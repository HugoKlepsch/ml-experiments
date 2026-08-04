"""Tests for vectorised collection.

The whole safety argument for `collect_batch` is one property: **playing N
episodes together must store exactly what playing them one at a time would.**
`--envs` is advertised as a pure throughput knob, so if that property does not
hold it is silently changing what a run learns from, and every result in the
repo becomes conditional on a setting nobody thinks of as an experiment.

So the tests here are mostly differential: run it both ways, compare. That
catches the failure modes that matter -- an agent drawing from the wrong rng, an
observation built at the wrong tick, a shared opponent leaking state between
episodes -- none of which would show up as a crash.
"""

import random
import unittest

import numpy as np
import torch

from agents.dqn import DQNAgent, MLP, ReplayBuffer
from core.agent import Agent, RandomAgent, WeightedAgent
from core.game import NO_ACTION
from core.registry import load_agent, make_game
from train.dqn import collect_batch, collect_episode, train_dqn


class RecordingBuffer:
    """Stores every transition in full, so two runs can be compared exactly."""

    def __init__(self):
        self.rows = []

    def add(self, obs, action, reward, next_obs, done, legal_next, nsteps=1):
        self.rows.append((
            tuple(obs), action, round(float(reward), 9), tuple(next_obs),
            bool(done), tuple(legal_next), nsteps,
        ))

    @property
    def size(self):
        return len(self.rows)


class CountingAgent(Agent):
    """Cycles through legal actions using only its own rng.

    Deliberately does *not* override `act_batch`, so it exercises the fallback
    path in the collector as well as being trivially deterministic per episode.
    """

    name = "counting"

    def act(self, game, state, player, rng):
        options = game.legal_actions(state, player)
        return rng.choice(options) if options else NO_ACTION


def serial(game, learner, opponent, seeds, seats, **kwargs):
    buffer = RecordingBuffer()
    scores = [
        collect_episode(game, learner, opponent, seed, buffer, seat, **kwargs)
        for seed, seat in zip(seeds, seats)
    ]
    return scores, buffer.rows


def batched(game, learner, opponent, seeds, seats, **kwargs):
    buffer = RecordingBuffer()
    scores = collect_batch(game, learner, opponent, seeds, buffer, seats, **kwargs)
    return scores, buffer.rows


class TestBatchedMatchesSerial(unittest.TestCase):
    """The transitions may be interleaved differently, but the *set* must match."""

    def check(self, game, learner, opponent="random", seeds=(11, 12, 13, 14),
              seats=None, **kwargs):
        seats = seats or [i % game.num_players for i in range(len(seeds))]
        want_scores, want_rows = serial(game, learner, opponent, seeds, seats, **kwargs)
        got_scores, got_rows = batched(game, learner, opponent, seeds, seats, **kwargs)

        self.assertEqual(got_scores, want_scores)
        self.assertEqual(len(got_rows), len(want_rows))
        # Concurrent play emits transitions in tick order across episodes rather
        # than episode by episode, so compare as multisets.
        self.assertCountEqual(got_rows, want_rows)
        return want_rows

    def test_snake_two_players(self):
        rows = self.check(make_game("snake", num_players=2), CountingAgent())
        self.assertGreater(len(rows), 20, "expected a non-trivial number of transitions")

    def test_snake_solo(self):
        self.check(make_game("snake", num_players=1), CountingAgent())

    def test_snake_four_players(self):
        """More seats means more opponents sharing one instance in the batch."""
        self.check(make_game("snake", num_players=4), CountingAgent(),
                   seeds=(21, 22, 23, 24, 25, 26))

    def test_turn_based_perfect_information(self):
        self.check(make_game("connect4"), CountingAgent())

    def test_turn_based_hidden_information(self):
        """Kuhn also has chance, so the game rng has to stay per-episode."""
        self.check(make_game("kuhn"), CountingAgent(), seeds=(31, 32, 33, 34, 35))

    def test_single_player_no_opponent(self):
        self.check(make_game("2048"), CountingAgent(), seeds=(41, 42))

    def test_with_n_step(self):
        for n_step in (1, 2, 3, 5):
            with self.subTest(n_step=n_step):
                self.check(make_game("snake", num_players=2), CountingAgent(),
                           n_step=n_step, gamma=0.9)

    def test_against_a_trained_opponent(self):
        """A shared opponent instance must not leak anything between episodes."""
        self.check(make_game("snake", num_players=2), CountingAgent(),
                   opponent="ga")

    def test_with_a_dqn_learner_greedy(self):
        """epsilon=0: no rng draws, so this isolates the batched forward pass."""
        game = make_game("snake", num_players=2)
        torch.manual_seed(0)
        learner = DQNAgent(MLP([game.obs_size, 8, game.num_actions]), epsilon=0.0)
        self.check(game, learner)

    def test_with_a_dqn_learner_exploring(self):
        """epsilon>0: `act_batch` must consume each episode's rng identically."""
        game = make_game("snake", num_players=2)
        torch.manual_seed(1)
        learner = DQNAgent(MLP([game.obs_size, 8, game.num_actions]), epsilon=0.35)
        self.check(game, learner)

    def test_with_a_conv_dqn_learner(self):
        game = make_game("snake", num_players=2)
        size, grid = game.obs_spec("planes")
        torch.manual_seed(2)
        from agents.dqn import ConvNet
        net = ConvNet(grid, size - grid[0] * grid[1] * grid[2], game.num_actions,
                      channels=4, blocks=1, reduce=2, hidden=8)
        self.check(game, DQNAgent(net, epsilon=0.2, encoding="planes"))

    def test_seats_rotate_independently_of_batching(self):
        """The learner sits in a different seat per episode inside one batch,
        which is the case where grouping by seat instead of by agent breaks."""
        self.check(make_game("snake", num_players=2), CountingAgent(),
                   seeds=(51, 52, 53, 54), seats=[0, 1, 1, 0])

    def test_a_batch_of_one_is_the_serial_path(self):
        self.check(make_game("snake", num_players=2), CountingAgent(), seeds=(61,))


class TestActBatch(unittest.TestCase):
    def test_default_act_batch_matches_act(self):
        game = make_game("snake", num_players=2)
        state = game.reset(random.Random(0))
        agent = RandomAgent()

        want = [agent.act(game, state, 0, random.Random(7)) for _ in range(4)]
        got = agent.act_batch(game, [state] * 4, [0] * 4,
                              [random.Random(7) for _ in range(4)])
        self.assertEqual(got, want)

    def test_dqn_act_batch_matches_act_when_greedy(self):
        game = make_game("snake", num_players=2)
        torch.manual_seed(3)
        agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]), epsilon=0.0)

        states, rng = [], random.Random(0)
        state = game.reset(rng)
        for _ in range(5):
            states.append(state)
            state = game.step(state, (0, 1), rng)

        want = [agent.act(game, s, 0, random.Random(0)) for s in states]
        got = agent.act_batch(game, states, [0] * len(states),
                              [random.Random(0) for _ in states])
        self.assertEqual(got, want)

    def test_dqn_act_batch_matches_act_when_exploring(self):
        game = make_game("snake", num_players=2)
        torch.manual_seed(4)
        agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]), epsilon=0.5)
        state = game.reset(random.Random(0))

        # Distinct rng streams per row, exactly as the collector supplies them.
        want = [agent.act(game, state, 0, random.Random(k)) for k in range(6)]
        got = agent.act_batch(game, [state] * 6, [0] * 6,
                              [random.Random(k) for k in range(6)])
        self.assertEqual(got, want)

    def test_supplied_observations_are_used(self):
        """The collector passes observations it already built. If they were
        ignored, the saving that motivates the whole batch would vanish -- and
        if they were used wrongly, the agent would decide on the wrong position."""
        game = make_game("snake", num_players=2)
        torch.manual_seed(5)
        agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]), epsilon=0.0)
        state = game.reset(random.Random(0))

        real = game.observe(state, 0)
        # An observation that is not the one this state would produce; the agent
        # must act on what it was handed.
        fake = [-x for x in real]
        on_real = agent.act_batch(game, [state], [0], [random.Random(0)],
                                  observations=[real])
        on_fake = agent.act_batch(game, [state], [0], [random.Random(0)],
                                  observations=[fake])
        self.assertEqual(on_real, agent.act_batch(game, [state], [0],
                                                  [random.Random(0)]))
        # Not a guarantee for every net, but for this one the two differ, which
        # is what shows the argument is actually being read.
        self.assertNotEqual(on_real, on_fake)

    def test_illegal_actions_are_never_chosen_in_a_batch(self):
        game = make_game("snake", num_players=2)
        torch.manual_seed(6)
        agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]), epsilon=0.3)

        states, rng = [], random.Random(1)
        state = game.reset(rng)
        for _ in range(8):
            states.append(state)
            state = game.step(state, (2, 3), rng)

        picks = agent.act_batch(game, states, [0] * len(states),
                                [random.Random(k) for k in range(len(states))])
        for state, pick in zip(states, picks):
            self.assertIn(pick, game.legal_actions(state, 0))


class TestCollectBatchContract(unittest.TestCase):
    def test_mismatched_seeds_and_seats_are_rejected(self):
        game = make_game("snake", num_players=2)
        with self.assertRaises(ValueError):
            collect_batch(game, CountingAgent(), "random", [1, 2],
                          RecordingBuffer(), [0])

    def test_every_episode_reports_a_score(self):
        game = make_game("snake", num_players=2)
        scores = collect_batch(game, CountingAgent(), "random",
                               [1, 2, 3], RecordingBuffer(), [0, 1, 0])
        self.assertEqual(len(scores), 3)

    def test_empty_batch_does_nothing(self):
        game = make_game("snake", num_players=2)
        buffer = RecordingBuffer()
        self.assertEqual(collect_batch(game, CountingAgent(), "random", [], buffer, []), [])
        self.assertEqual(buffer.rows, [])


class TestEnvsFlag(unittest.TestCase):
    def test_envs_does_not_change_the_episode_budget(self):
        """`--envs` is a throughput knob. If it changed how many episodes or
        gradient steps a run did, every sweep against it would be measuring two
        things at once."""
        for envs in (1, 3, 8):
            with self.subTest(envs=envs):
                result = train_dqn(
                    "snake", players=2, envs=envs, episodes=10, hidden=8,
                    buffer=500, warmup=10_000,  # warmup high: skip all updates
                    eval_every=10, eval_games=1, save=False, quiet=True,
                )
                self.assertEqual(result["history"][-1]["episode"], 10)

    def test_envs_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            train_dqn("snake", players=2, envs=0, episodes=1, save=False, quiet=True)

    def test_a_short_vectorised_run_trains(self):
        result = train_dqn(
            "snake", players=2, envs=8, episodes=48, hidden=8, buffer=4000,
            warmup=100, updates_per_episode=2, eval_every=48, eval_games=2,
            save=False, quiet=True,
        )
        self.assertTrue(result["history"])
        self.assertEqual(result["history"][-1]["episode"], 48)

    def test_uneven_batches_still_total_the_episode_count(self):
        """7 episodes at 4 at a time is 4 then 3, and the short tail must not be
        dropped or rounded up."""
        result = train_dqn(
            "snake", players=2, envs=4, episodes=7, hidden=8, buffer=500,
            warmup=10_000, eval_every=1, eval_games=1, save=False, quiet=True,
        )
        self.assertEqual(result["history"][-1]["episode"], 7)


if __name__ == "__main__":
    unittest.main()
