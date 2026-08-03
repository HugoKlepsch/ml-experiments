"""Tests for the programmatic trainer entry points and the sweep helpers.

Training runs here are deliberately tiny -- these check plumbing and contracts,
not that anything learns.
"""

import unittest

from train.dqn import DEFAULTS as DQN_DEFAULTS, train_dqn
from train.ga import DEFAULTS as GA_DEFAULTS, train_ga
from train.sweep import evaluate_model, round_robin, separable, sweep


class TestTrainerEntryPoints(unittest.TestCase):
    def test_ga_returns_a_result_without_saving(self):
        result = train_ga("snake", players=2, generations=2, population=6,
                          games=3, holdout_games=3, save=False, quiet=True)
        self.assertIsNone(result["path"])
        self.assertEqual(len(result["weights"]), 7)
        self.assertEqual(len(result["history"]), 2)
        self.assertIn("holdout_score", result)

    def test_dqn_returns_a_result_without_saving(self):
        result = train_dqn("snake", players=2, episodes=60, hidden=8,
                           eval_every=60, eval_games=2, warmup=20,
                           save=False, quiet=True)
        self.assertIsNone(result["path"])
        self.assertTrue(result["history"])
        self.assertIn("eval_score", result)

    def test_unknown_parameters_are_rejected(self):
        # Silently ignoring a typo would mean sweeping a parameter that never
        # reached the trainer, and reporting the resulting noise as a finding.
        with self.assertRaises(TypeError) as caught:
            train_ga("snake", populaton=10)
        self.assertIn("populaton", str(caught.exception))

        with self.assertRaises(TypeError):
            train_dqn("snake", hiden=32)

    def test_defaults_cover_every_documented_knob(self):
        for required in ("name", "players", "seed", "save", "quiet"):
            self.assertIn(required, GA_DEFAULTS)
            self.assertIn(required, DQN_DEFAULTS)


class TestEvaluateModel(unittest.TestCase):
    def test_reports_score_and_win_rate_with_intervals(self):
        result = evaluate_model("snake", "ga", games=40, players=2)
        self.assertEqual(result["games"], 40)
        self.assertLessEqual(result["score_low"], result["mean_score"])
        self.assertGreaterEqual(result["score_high"], result["mean_score"])
        self.assertLessEqual(result["win_low"], result["win_rate"])
        self.assertGreaterEqual(result["win_high"], result["win_rate"])

    def test_single_player_game_is_supported(self):
        result = evaluate_model("2048", "ga", games=8)
        self.assertEqual(result["games"], 8)
        self.assertGreater(result["mean_score"], 0)

    def test_a_trained_model_beats_random(self):
        trained = evaluate_model("snake", "ga", games=60, players=2)
        baseline = evaluate_model("snake", "random", games=60, players=2)
        self.assertGreater(trained["mean_score"], baseline["mean_score"])


class TestSweep(unittest.TestCase):
    def test_produces_one_row_per_value(self):
        rows = sweep("dqn", "snake", "hidden", [4, 8], players=2, prefix="_test",
                     episodes=40, eval_every=40, eval_games=2, warmup=10,
                     arena_games=20, verbose=False)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["hidden"] for r in rows], [4, 8])
        self.assertEqual([r["model"] for r in rows], ["_test-4", "_test-8"])
        for row in rows:
            self.assertIn("mean_score", row)
            self.assertIn("win_rate", row)

    def test_arena_games_does_not_capture_the_trainers_eval_games(self):
        """Distinct names matter: both used to be called eval_games."""
        rows = sweep("dqn", "snake", "hidden", [4], players=2, prefix="_test",
                     episodes=40, eval_every=40, eval_games=3, warmup=10,
                     arena_games=20, verbose=False)
        self.assertEqual(rows[0]["games"], 20)          # the arena measurement
        self.assertEqual(rows[0]["model"], "_test-4")

    def test_typos_in_swept_parameters_are_rejected(self):
        with self.assertRaises(TypeError):
            sweep("dqn", "snake", "hiddn", [4], players=2, verbose=False,
                  episodes=20, arena_games=10)

    def test_unknown_trainer_is_rejected(self):
        with self.assertRaises(ValueError):
            sweep("xgboost", "snake", "hidden", [4], verbose=False)

    def test_output_lands_in_the_gitignored_sweeps_directory(self):
        from core.registry import find_model, list_agents

        sweep("dqn", "snake", "hidden", [4], players=2, prefix="_test",
              episodes=40, eval_every=40, eval_games=2, warmup=10,
              arena_games=10, verbose=False)

        path = find_model("snake", "_test-4")
        self.assertIsNotNone(path)
        # Kept out of version control, but still discoverable by the arena
        # and the match viewer.
        self.assertEqual(path.parent.name, "sweeps")
        self.assertIn("_test-4", list_agents("snake"))

    def tearDown(self):
        from core.registry import model_path
        for name in ("_test-4", "_test-8"):
            for sweep_flag in (True, False):
                path = model_path("snake", name, sweep=sweep_flag)
                if path.exists():
                    path.unlink()


class TestRoundRobin(unittest.TestCase):
    def test_matrix_is_antisymmetric(self):
        result = round_robin("snake", ["ga", "random"], games=40, players=2)
        forward = result["win_rate"]["ga"]["random"]
        backward = result["win_rate"]["random"]["ga"]
        self.assertAlmostEqual(forward + backward, 1.0, places=6)

    def test_diagonal_is_undefined(self):
        result = round_robin("snake", ["ga", "random"], games=20, players=2)
        self.assertNotEqual(result["win_rate"]["ga"]["ga"],
                            result["win_rate"]["ga"]["ga"])  # nan != nan

    def test_trained_model_beats_random_convincingly(self):
        result = round_robin("snake", ["ga", "random"], games=100, players=2)
        low, _high = result["interval"]["ga"]["random"]
        self.assertGreater(low, 0.5)


class TestSeparable(unittest.TestCase):
    def test_overlapping_intervals_are_not_separable(self):
        self.assertFalse(separable(0.45, 0.55, 0.48, 0.58))

    def test_disjoint_intervals_are_separable(self):
        self.assertTrue(separable(0.10, 0.20, 0.80, 0.90))
        self.assertTrue(separable(0.80, 0.90, 0.10, 0.20))

    def test_touching_intervals_are_not_separable(self):
        self.assertFalse(separable(0.40, 0.50, 0.50, 0.60))


if __name__ == "__main__":
    unittest.main()
