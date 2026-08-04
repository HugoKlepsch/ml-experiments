"""Tests for n-step returns and the convolutional path.

Both of these fail *quietly*. An n-step return with the wrong discount, or a
grid flattened in the wrong order, still trains — it just trains worse, and the
only symptom is a number in a sweep that looks like ordinary run-to-run noise.
So the arithmetic is checked directly here rather than inferred from whether a
short training run improved.
"""

import json
import random
import unittest

import numpy as np
import torch

from agents.dqn import ConvNet, DQNAgent, MLP, ReplayBuffer, build_net
from core.game import Game
from core.registry import load_agent, make_game, model_path
from train.dqn import build_for, collect_episode, train_dqn
from types import SimpleNamespace


class StubBuffer:
    """Records `add` calls instead of storing them, so a test can read them."""

    def __init__(self):
        self.rows = []

    def add(self, obs, action, reward, next_obs, done, legal_next, nsteps=1):
        self.rows.append({
            "action": action, "reward": reward, "done": done,
            "legal": list(legal_next), "nsteps": nsteps,
        })


class CountingGame(Game):
    """One player, `length` decisions, reward 1.0 per step and nothing else.

    Purpose-built so the correct n-step return is something you can work out in
    your head: with all rewards 1, an n-step return is 1 + g + ... + g^(n-1).
    """

    name = "counting"
    num_players = 1
    action_names = ("only",)
    feature_names = ("x",)

    def __init__(self, length=6):
        self.length = length
        self.obs_size = 1

    def reset(self, rng):
        return 0

    def legal_actions(self, state, player):
        return [] if self.is_terminal(state) else [0]

    def step(self, state, actions, rng):
        return state + 1

    def is_terminal(self, state):
        return state >= self.length

    def scores(self, state):
        return (float(state),)

    def action_features(self, state, player, action):
        return (1.0,)

    def observe(self, state, player, encoding=None):
        return [float(state)]

    def render(self, state):
        return {"kind": "counting", "step": state}


class ScriptedAgent:
    """Always takes action 0. Not a DQNAgent, so it has no `encoding`."""

    name = "scripted"

    def act(self, game, state, player, rng):
        options = game.legal_actions(state, player)
        return options[0] if options else -1

    def reset(self):
        pass


class TestNStepReturns(unittest.TestCase):
    def collect(self, n_step, gamma=0.5, length=6):
        game = CountingGame(length)
        buffer = StubBuffer()
        collect_episode(game, ScriptedAgent(), "random", 1, buffer, 0,
                        n_step=n_step, gamma=gamma)
        return buffer.rows

    def test_one_step_stores_the_raw_reward(self):
        rows = self.collect(n_step=1)
        self.assertTrue(all(r["nsteps"] == 1 for r in rows))
        for row in rows:
            self.assertAlmostEqual(row["reward"], 1.0, places=6)

    def test_n_step_return_is_the_discounted_sum(self):
        """With every reward 1 and gamma 0.5, a 3-step return is 1 + .5 + .25."""
        rows = self.collect(n_step=3, gamma=0.5)
        full = [r for r in rows if r["nsteps"] == 3]
        self.assertTrue(full, "expected some full-length windows")
        for row in full:
            self.assertAlmostEqual(row["reward"], 1.75, places=6)

    def test_transitions_shortened_at_the_end_report_their_true_length(self):
        """The last few decisions have fewer than n steps left. They are still
        worth storing -- they are the ones nearest the outcome -- but they must
        say how long they are so the bootstrap is discounted correctly."""
        rows = self.collect(n_step=3, gamma=0.5, length=6)
        tail = [r for r in rows if r["done"]]
        self.assertEqual(sorted(r["nsteps"] for r in tail), [1, 2, 3])
        by_length = {r["nsteps"]: r["reward"] for r in tail}
        self.assertAlmostEqual(by_length[1], 1.0, places=6)
        self.assertAlmostEqual(by_length[2], 1.5, places=6)
        self.assertAlmostEqual(by_length[3], 1.75, places=6)

    def test_every_decision_is_stored_exactly_once(self):
        for n_step in (1, 2, 3, 5):
            with self.subTest(n_step=n_step):
                rows = self.collect(n_step=n_step, length=6)
                self.assertEqual(len(rows), 6)

    def test_a_window_longer_than_the_episode_still_terminates(self):
        rows = self.collect(n_step=99, length=4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["done"] for r in rows))

    def test_only_the_final_transitions_are_marked_done(self):
        rows = self.collect(n_step=2, length=6)
        # Everything emitted mid-episode bootstraps; only the flush at the end
        # is terminal, and it covers as many rows as the window held.
        self.assertEqual(sum(r["done"] for r in rows), 2)

    def test_n_step_matches_a_hand_computed_return_on_snake(self):
        """The same property on a real game, where rewards actually vary."""
        game = make_game("snake", num_players=1)
        gamma, n_step = 0.9, 3

        one = StubBuffer()
        collect_episode(game, ScriptedAgent(), "random", 5, one, 0,
                        n_step=1, gamma=gamma)
        many = StubBuffer()
        collect_episode(game, ScriptedAgent(), "random", 5, many, 0,
                        n_step=n_step, gamma=gamma)

        # Same episode, so the per-decision rewards are identical; the n-step
        # return at decision t must be the discounted sum of the 1-step ones.
        rewards = [r["reward"] for r in one.rows]
        self.assertEqual(len(one.rows), len(many.rows))
        for t, row in enumerate(many.rows):
            window = rewards[t:t + row["nsteps"]]
            expected = sum(gamma ** k * r for k, r in enumerate(window))
            self.assertAlmostEqual(row["reward"], expected, places=5)


class TestPlanesEncoding(unittest.TestCase):
    def setUp(self):
        self.game = make_game("snake", num_players=2)
        self.state = self.game.reset(random.Random(0))

    def test_spec_matches_what_observe_returns(self):
        for encoding in self.game.encodings:
            with self.subTest(encoding=encoding):
                size, _ = self.game.obs_spec(encoding)
                self.assertEqual(len(self.game.observe(self.state, 0, encoding)), size)

    def test_default_encoding_is_unchanged(self):
        """Everything already trained and committed used the flat vector, so
        `observe` with no encoding has to keep returning exactly that."""
        self.assertEqual(
            self.game.observe(self.state, 0),
            self.game.observe(self.state, 0, "flat"),
        )
        self.assertEqual(self.game.obs_spec(), (self.game.obs_size, None))

    def test_planes_append_the_flat_vector(self):
        planes = self.game.observe(self.state, 0, "planes")
        flat = self.game.observe(self.state, 0, "flat")
        self.assertEqual(planes[-len(flat):], flat)

    def test_grid_is_binary_and_laid_out_in_c_order(self):
        planes = self.game.observe(self.state, 0, "planes")
        _, (channels, height, width) = self.game.obs_spec("planes")
        grid = np.array(planes[:channels * height * width]).reshape(
            channels, height, width
        )
        self.assertEqual(set(np.unique(grid)), {0.0, 1.0})

        # The head plane must have a 1 exactly where the head is, at [y][x].
        head_x, head_y = self.state.bodies[0][0]
        self.assertEqual(grid[0].sum(), 1.0)
        self.assertEqual(grid[0][head_y][head_x], 1.0)

    def test_each_player_sees_itself_in_the_own_planes(self):
        """Planes are relative to the observer. If they were absolute, seat
        rotation in the arena would silently hand a model the wrong snake."""
        _, (_, height, width) = self.game.obs_spec("planes")
        cells = height * width
        for player in (0, 1):
            planes = self.game.observe(self.state, player, "planes")
            head_x, head_y = self.state.bodies[player][0]
            self.assertEqual(planes[head_y * width + head_x], 1.0)

    def test_food_plane_matches_the_state(self):
        planes = self.game.observe(self.state, 0, "planes")
        _, (_, height, width) = self.game.obs_spec("planes")
        cells = height * width
        food = planes[6 * cells:7 * cells]
        self.assertEqual(sum(food), len(self.state.food))
        for x, y in self.state.food:
            self.assertEqual(food[y * width + x], 1.0)

    def test_on_board_plane_is_all_ones(self):
        """It exists so that zero-padding in a convolution reads as a wall. If
        it were ever partly zero it would read as a wall in mid-board."""
        planes = self.game.observe(self.state, 0, "planes")
        _, (_, height, width) = self.game.obs_spec("planes")
        cells = height * width
        self.assertEqual(sum(planes[7 * cells:8 * cells]), cells)

    def test_dead_snakes_leave_the_planes(self):
        """`step` stops updating a dead snake's body and `_obstacles` ignores
        it, so a stale body in the planes would be an obstacle that is not
        there."""
        state = self.state
        alive = list(state.alive)
        alive[1] = False
        state = type(state)(**{**state.__dict__, "alive": tuple(alive)})

        planes = self.game.observe(state, 0, "planes")
        _, (_, height, width) = self.game.obs_spec("planes")
        cells = height * width
        for plane in (3, 4, 5):
            self.assertEqual(sum(planes[plane * cells:(plane + 1) * cells]), 0.0)

    def test_unknown_encoding_is_rejected(self):
        with self.assertRaises(ValueError):
            self.game.observe(self.state, 0, "pixels")
        with self.assertRaises(ValueError):
            self.game.obs_spec("pixels")

    def test_games_without_a_grid_report_none(self):
        for name in ("2048", "connect4", "kuhn"):
            with self.subTest(name=name):
                game = make_game(name)
                self.assertEqual(game.encodings, ("flat",))
                self.assertEqual(game.obs_spec(), (game.obs_size, None))


class TestConvNet(unittest.TestCase):
    def net(self, **kwargs):
        options = {"grid": (3, 5, 5), "scalars": 4, "n_actions": 3,
                   "channels": 8, "blocks": 2, "reduce": 4, "hidden": 16}
        options.update(kwargs)
        return ConvNet(**options)

    def test_output_shape(self):
        net = self.net()
        out = net(torch.randn(7, 3 * 5 * 5 + 4))
        self.assertEqual(tuple(out.shape), (7, 3))

    def test_every_parameter_receives_a_gradient(self):
        net = self.net()
        x = torch.randn(4, 3 * 5 * 5 + 4)
        net(x).sum().backward()
        for name, param in net.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{name} got no gradient")
            self.assertGreater(float(param.grad.abs().sum()), 0.0, msg=name)

    def test_scalar_channel_reaches_the_output(self):
        """It would be easy to slice the input wrongly and drop the scalars
        entirely; the net would still train, just blind to them."""
        net = self.net()
        base = torch.zeros(1, 3 * 5 * 5 + 4)
        changed = base.clone()
        changed[0, -1] = 5.0
        with torch.no_grad():
            self.assertFalse(torch.allclose(net(base), net(changed)))

    def test_grid_channel_reaches_the_output(self):
        net = self.net()
        base = torch.zeros(1, 3 * 5 * 5 + 4)
        changed = base.clone()
        changed[0, 12] = 1.0
        with torch.no_grad():
            self.assertFalse(torch.allclose(net(base), net(changed)))

    def test_position_matters(self):
        """No pooling, so the same mark in two places must give two answers.
        A stray global pool would make these identical and cost the net every
        spatial fact it has."""
        net = self.net()
        left, right = torch.zeros(1, 3 * 5 * 5 + 4), torch.zeros(1, 3 * 5 * 5 + 4)
        left[0, 0] = 1.0    # top-left of plane 0
        right[0, 24] = 1.0  # bottom-right of plane 0
        with torch.no_grad():
            self.assertFalse(torch.allclose(net(left), net(right)))

    def test_head_is_linear(self):
        net = self.net()
        with torch.no_grad():
            net.head[-1].weight.zero_()
            net.head[-1].bias.fill_(-1.0)
            out = net(torch.randn(2, 3 * 5 * 5 + 4))
        torch.testing.assert_close(out, torch.full((2, 3), -1.0))

    def test_wrong_input_width_is_rejected_with_a_useful_message(self):
        net = self.net()
        with self.assertRaisesRegex(ValueError, "encoding"):
            net(torch.randn(2, 10))

    def test_reduce_conv_shrinks_the_head(self):
        big = self.net(reduce=8)
        small = self.net(reduce=2)
        self.assertLess(
            sum(p.numel() for p in small.head.parameters()),
            sum(p.numel() for p in big.head.parameters()),
        )


class TestConvSerialisation(unittest.TestCase):
    def test_round_trip_through_json_preserves_predictions(self):
        game = make_game("snake", num_players=2)
        size, grid = game.obs_spec("planes")
        scalars = size - grid[0] * grid[1] * grid[2]
        net = ConvNet(grid, scalars, game.num_actions, channels=8, blocks=1,
                      reduce=2, hidden=8)
        agent = DQNAgent(net, name="c", encoding="planes")

        payload = json.loads(json.dumps(agent.to_payload()))  # as it hits disk
        restored = DQNAgent.from_payload(payload, name="c")

        self.assertEqual(restored.encoding, "planes")
        self.assertIsInstance(restored.net, ConvNet)
        x = torch.randn(2, size)
        with torch.no_grad():
            torch.testing.assert_close(net(x), restored.net(x))

    def test_payload_without_an_arch_loads_as_an_mlp(self):
        """The committed models predate `arch`. They must keep loading, or the
        README's arena numbers stop being reproducible."""
        payload = {
            "kind": "dqn",
            "sizes": [4, 6, 3],
            "state_dict": {
                k: v.tolist() for k, v in MLP([4, 6, 3]).state_dict().items()
            },
        }
        agent = DQNAgent.from_payload(payload, name="legacy")
        self.assertIsInstance(agent.net, MLP)
        self.assertIsNone(agent.encoding)

    def test_build_net_rejects_an_unknown_kind(self):
        with self.assertRaises(ValueError):
            build_net({"kind": "transformer"})

    def test_arch_is_exactly_the_constructor_arguments(self):
        net = ConvNet((2, 4, 4), 3, 5, channels=8, blocks=1, reduce=2, hidden=8)
        rebuilt = build_net(net.arch)
        self.assertEqual(rebuilt.arch, net.arch)


class TestArchitectureSelection(unittest.TestCase):
    def args(self, **kwargs):
        options = {"encoding": "flat", "hidden": 16, "channels": 8, "blocks": 1,
                   "reduce": 2}
        options.update(kwargs)
        return SimpleNamespace(**options)

    def test_flat_encoding_builds_an_mlp_with_float32_storage(self):
        game = make_game("snake", num_players=2)
        make, size, dtype = build_for(game, self.args())
        self.assertIsInstance(make(), MLP)
        self.assertEqual(size, game.obs_size)
        self.assertEqual(dtype, np.float32)

    def test_grid_encoding_builds_a_conv_with_half_precision_storage(self):
        game = make_game("snake", num_players=2)
        make, size, dtype = build_for(game, self.args(encoding="planes"))
        net = make()
        self.assertIsInstance(net, ConvNet)
        self.assertEqual(size, game.planes_size)
        self.assertEqual(net.scalars, game.obs_size)
        self.assertEqual(dtype, np.float16)

    def test_the_factory_is_called_fresh_each_time(self):
        """`run` builds the online and target nets separately and then copies
        one into the other. If the factory returned a shared module the target
        would track the online net exactly and stop being a target at all."""
        game = make_game("snake", num_players=2)
        make, _, _ = build_for(game, self.args())
        self.assertIsNot(make(), make())


class TestEndToEnd(unittest.TestCase):
    def tearDown(self):
        # Written with sweep=True, so it lands in the gitignored sweeps/
        # directory -- but it would still show up in the arena and the match
        # viewer as though it were a real model.
        written = model_path("snake", "test-conv", sweep=True)
        written.unlink(missing_ok=True)

    def test_a_short_conv_run_trains_and_reloads(self):
        result = train_dqn(
            "snake", players=2, encoding="planes", n_step=3, episodes=40,
            hidden=8, channels=8, blocks=1, reduce=2, buffer=2000, warmup=50,
            eval_every=40, eval_games=2, name="test-conv", save=True,
            sweep=True, quiet=True,
        )
        self.assertGreater(result["params"], 0)

        agent = load_agent("snake", "test-conv")
        self.assertEqual(agent.encoding, "planes")
        self.assertIsInstance(agent.net, ConvNet)

        # And it can actually take a turn against a flat-encoded opponent off a
        # single game object, which is the whole point of per-agent encodings.
        game = make_game("snake", num_players=2)
        state = game.reset(random.Random(0))
        action = agent.act(game, state, 0, random.Random(0))
        self.assertIn(action, game.legal_actions(state, 0))

    def test_unknown_encoding_fails_before_training_starts(self):
        with self.assertRaisesRegex(ValueError, "encoding"):
            train_dqn("connect4", encoding="planes", episodes=1, quiet=True,
                      save=False)

    def test_n_step_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            train_dqn("snake", players=2, n_step=0, episodes=1, quiet=True,
                      save=False)


if __name__ == "__main__":
    unittest.main()
