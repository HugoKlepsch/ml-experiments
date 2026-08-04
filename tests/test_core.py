"""Tests for the runner, agents, arena statistics and the DQN's network."""

import json
import random
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from agents.dqn import MLP, DQNAgent, ReplayBuffer
from core.agent import RandomAgent, WeightedAgent
from core.arena import wilson_interval
from core.registry import find_model, list_agents, list_games, load_agent, make_game
from core.runner import play_episode


class TestRunner(unittest.TestCase):
    def test_same_agents_and_seed_reproduce_the_episode(self):
        game = make_game("2048")
        weights = [0.5] * len(game.feature_names)
        a = play_episode(game, [WeightedAgent(weights)], 42)
        b = play_episode(game, [WeightedAgent(weights)], 42)
        self.assertEqual(a.scores, b.scores)
        self.assertEqual(a.steps, b.steps)

    def test_different_seeds_diverge(self):
        game = make_game("2048")
        weights = [0.5] * len(game.feature_names)
        scores = {play_episode(game, [WeightedAgent(weights)], s).scores[0]
                  for s in range(8)}
        self.assertGreater(len(scores), 1)

    def test_wrong_agent_count_is_rejected(self):
        game = make_game("snake", num_players=2)
        with self.assertRaises(ValueError):
            play_episode(game, [RandomAgent()], 0)

    def test_recording_captures_every_step(self):
        game = make_game("snake", num_players=2)
        episode = play_episode(game, [RandomAgent(), RandomAgent()], 3, record=True)
        self.assertEqual(len(episode.frames), episode.steps + 1)
        self.assertIn("scores", episode.frames[0])
        self.assertIn("kind", episode.frames[0])

    def test_snake_episode_terminates(self):
        game = make_game("snake", num_players=2)
        episode = play_episode(game, [RandomAgent(), RandomAgent()], 7)
        self.assertTrue(episode.steps > 0)
        self.assertLessEqual(len(episode.winners), 2)


class TestAgents(unittest.TestCase):
    def test_weighted_agent_only_returns_legal_actions(self):
        game = make_game("snake", num_players=2)
        agent = WeightedAgent([1.0] * len(game.feature_names))
        rng = random.Random(0)
        state = game.reset(random.Random(0))
        for _ in range(20):
            if game.is_terminal(state):
                break
            actions = tuple(
                agent.act(game, state, i, rng) if game.legal_actions(state, i) else -1
                for i in range(2)
            )
            for i, a in enumerate(actions):
                if game.legal_actions(state, i):
                    self.assertIn(a, game.legal_actions(state, i))
            state = game.step(state, actions, rng)

    def test_weights_change_behaviour(self):
        """A weight vector that fears death should outlive one that seeks it."""
        game = make_game("snake", num_players=1)
        names = game.feature_names
        cautious = [-5.0 if n == "dies" else 0.0 for n in names]
        reckless = [+5.0 if n == "dies" else 0.0 for n in names]
        good = play_episode(game, [WeightedAgent(cautious)], 11).steps
        bad = play_episode(game, [WeightedAgent(reckless)], 11).steps
        self.assertGreater(good, bad)


class TestRegistry(unittest.TestCase):
    def test_games_are_discoverable(self):
        self.assertEqual(list_games(), ["2048", "connect4", "kuhn", "snake"])

    def test_baselines_always_available(self):
        self.assertIn("random", list_agents("2048"))
        self.assertIn("first", list_agents("2048"))

    def test_unknown_agent_raises(self):
        with self.assertRaises(KeyError):
            load_agent("2048", "no-such-model")


class TestWilson(unittest.TestCase):
    def test_interval_brackets_the_estimate(self):
        rate, low, high = wilson_interval(55, 100)
        self.assertAlmostEqual(rate, 0.55)
        self.assertLess(low, 0.55)
        self.assertGreater(high, 0.55)

    def test_more_games_tighten_the_interval(self):
        _, low_small, high_small = wilson_interval(55, 100)
        _, low_big, high_big = wilson_interval(550, 1000)
        self.assertGreater(high_small - low_small, high_big - low_big)

    def test_extremes_stay_inside_zero_and_one(self):
        for successes, total in [(0, 20), (20, 20)]:
            _, low, high = wilson_interval(successes, total)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)


class TestNetwork(unittest.TestCase):
    def test_every_parameter_receives_a_gradient(self):
        """Autograd does the differentiating, so the thing worth checking is that
        the graph reaches all of it -- a detached tensor or a layer left out of
        the optimiser trains silently and badly."""
        net = MLP([5, 7, 4])
        x = torch.randn(6, 5)
        target = torch.randn(6, 4)

        ((net(x) - target) ** 2).mean().backward()
        for name, param in net.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{name} got no gradient")
            self.assertGreater(float(param.grad.abs().sum()), 0.0, msg=name)

    def test_head_is_linear(self):
        """A ReLU on the output would floor every Q-value at zero, which quietly
        breaks any game with negative rewards -- Kuhn poker loses chips."""
        net = MLP([4, 8, 3])
        self.assertIsInstance(net.net[-1], nn.Linear)
        with torch.no_grad():
            # Drive the head to a known negative value; a trailing ReLU would
            # clamp it to 0.
            net.net[-1].weight.zero_()
            net.net[-1].bias.fill_(-1.0)
            torch.testing.assert_close(net(torch.randn(2, 4)), torch.full((2, 3), -1.0))

    def test_network_can_fit_a_tiny_dataset(self):
        net = MLP([3, 16, 2])
        opt = torch.optim.Adam(net.parameters(), lr=0.02)
        x = torch.eye(3)
        y = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5]])
        with torch.no_grad():
            first = float(((net(x) - y) ** 2).mean())
        for _ in range(400):
            opt.zero_grad(set_to_none=True)
            ((net(x) - y) ** 2).mean().backward()
            opt.step()
        with torch.no_grad():
            self.assertLess(float(((net(x) - y) ** 2).mean()), first * 0.05)

    def test_huber_gradient_is_clipped(self):
        """The training step relies on smooth_l1_loss having beta=1, i.e. on its
        gradient being the residual clipped to [-1, 1]."""
        predicted = torch.tensor([0.5, 10.0, -10.0], requires_grad=True)
        F.smooth_l1_loss(predicted, torch.zeros(3), reduction="sum").backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([0.5, 1.0, -1.0]))

    def test_round_trip_through_json_preserves_predictions(self):
        net = MLP([4, 8, 3])
        agent = DQNAgent(net, name="x")
        payload = json.loads(json.dumps(agent.to_payload()))  # as it hits disk
        restored = DQNAgent.from_payload(payload, name="x")
        x = torch.randn(2, 4)
        with torch.no_grad():
            torch.testing.assert_close(net(x), restored.net(x))

    def test_committed_models_load(self):
        """models/ is committed and the arena numbers in the README come from it,
        so a change to the payload layout has to be caught here, not by someone
        wondering why a trained model suddenly plays like it is untrained."""
        loaded = 0
        for game_name in list_games():
            game = make_game(game_name)
            for name in list_agents(game_name):
                path = find_model(game_name, name)
                if path is None or json.loads(path.read_text()).get("kind") != "dqn":
                    continue
                agent = load_agent(game_name, name)
                with torch.no_grad():
                    q = agent.net(torch.zeros(1, game.obs_size))
                self.assertEqual(tuple(q.shape), (1, game.num_actions))
                loaded += 1
        self.assertGreater(loaded, 0, "no committed DQN models found to check")

    def test_dqn_masks_illegal_actions(self):
        game = make_game("snake", num_players=1)
        agent = DQNAgent(MLP([game.obs_size, 8, 4]))
        state = game.reset(random.Random(0))
        for _ in range(30):
            if game.is_terminal(state):
                break
            legal = game.legal_actions(state, 0)
            action = agent.act(game, state, 0, random.Random(0))
            self.assertIn(action, legal)
            state = game.step(state, (action,), random.Random(0))


class TestReplayBuffer(unittest.TestCase):
    def test_wraps_at_capacity(self):
        buffer = ReplayBuffer(4, obs_size=2)
        for i in range(10):
            buffer.add([i, i], 0, float(i), [i, i], False, [0, 1])
        self.assertEqual(buffer.size, 4)

    def test_sample_shapes(self):
        buffer = ReplayBuffer(16, obs_size=3)
        for i in range(16):
            buffer.add([i, 0, 1], i % 4, 1.0, [0, 1, 2], i % 2 == 0, [0, 2])
        obs, actions, rewards, next_obs, done, legal = buffer.sample(5)
        self.assertEqual(obs.shape, (5, 3))
        self.assertEqual(actions.shape, (5,))
        self.assertEqual(legal.shape, (5, 8))


if __name__ == "__main__":
    unittest.main()
