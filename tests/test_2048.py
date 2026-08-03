"""Rules tests for 2048."""

import random
import unittest

from games.g2048 import (
    DOWN, EMPTY_BOARD, LEFT, RIGHT, UP, Game2048, _slide_row, move, spawn,
)


def board(*rows):
    return tuple(tile for row in rows for tile in row)


class TestSlide(unittest.TestCase):
    def test_compacts_without_merging(self):
        self.assertEqual(_slide_row((0, 1, 0, 2)), ((1, 2, 0, 0), 0))

    def test_merges_equal_neighbours(self):
        self.assertEqual(_slide_row((1, 1, 0, 0)), ((2, 0, 0, 0), 4))

    def test_each_tile_merges_at_most_once(self):
        # 2,2,2 -> 4,2 and not 8. Merges resolve from the left.
        self.assertEqual(_slide_row((1, 1, 1, 0)), ((2, 1, 0, 0), 4))

    def test_two_independent_merges(self):
        self.assertEqual(_slide_row((1, 1, 1, 1)), ((2, 2, 0, 0), 8))

    def test_unequal_tiles_do_not_merge(self):
        self.assertEqual(_slide_row((1, 2, 3, 4)), ((1, 2, 3, 4), 0))

    def test_score_is_the_face_value_of_the_new_tile(self):
        self.assertEqual(_slide_row((3, 3, 0, 0)), ((4, 0, 0, 0), 16))


class TestMove(unittest.TestCase):
    def setUp(self):
        self.b = board((1, 1, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 3))

    def test_left(self):
        expected = board((2, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0), (3, 0, 0, 0))
        self.assertEqual(move(self.b, LEFT)[:2], (expected, 4))

    def test_right(self):
        expected = board((0, 0, 0, 2), (0, 0, 0, 2), (0, 0, 0, 0), (0, 0, 0, 3))
        self.assertEqual(move(self.b, RIGHT)[:2], (expected, 4))

    def test_up(self):
        expected = board((1, 1, 0, 3), (0, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(move(self.b, UP)[:2], (expected, 0))

    def test_down(self):
        expected = board((0, 0, 0, 0), (0, 0, 0, 0), (0, 1, 0, 0), (1, 2, 0, 3))
        self.assertEqual(move(self.b, DOWN)[:2], (expected, 0))

    def test_reports_no_change_when_blocked(self):
        # A full row of distinct tiles cannot shift or merge horizontally, and
        # it is already at the top, so only DOWN does anything.
        packed = board((1, 2, 3, 4), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.assertFalse(move(packed, LEFT)[2])
        self.assertFalse(move(packed, RIGHT)[2])
        self.assertFalse(move(packed, UP)[2])
        self.assertTrue(move(packed, DOWN)[2])


class TestGameInterface(unittest.TestCase):
    def setUp(self):
        self.game = Game2048()

    def test_new_game_places_exactly_two_tiles(self):
        state = self.game.reset(random.Random(0))
        self.assertEqual(sum(1 for t in state.board if t), 2)
        self.assertTrue(all(t in (0, 1, 2) for t in state.board))

    def test_full_alternating_board_is_terminal(self):
        from games.g2048 import State2048
        stuck = State2048(
            board((1, 2, 1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 2, 1)), 0
        )
        self.assertEqual(self.game.legal_actions(stuck, 0), [])
        self.assertTrue(self.game.is_terminal(stuck))

    def test_full_board_with_a_pair_is_not_terminal(self):
        from games.g2048 import State2048
        live = State2048(
            board((1, 1, 1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 2, 1)), 0
        )
        self.assertFalse(self.game.is_terminal(live))

    def test_spawn_on_full_board_is_a_noop(self):
        full = board((1, 2, 1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 1, 2, 1))
        self.assertEqual(spawn(full, random.Random(0)), full)

    def test_spawn_fills_one_empty_cell(self):
        self.assertEqual(
            sum(1 for t in spawn(EMPTY_BOARD, random.Random(7)) if t), 1
        )

    def test_step_rejects_an_illegal_move(self):
        from games.g2048 import State2048
        packed = State2048(
            board((1, 2, 3, 4), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), 0
        )
        with self.assertRaises(ValueError):
            self.game.step(packed, (LEFT,), random.Random(0))

    def test_feature_and_observation_widths_match_declarations(self):
        state = self.game.reset(random.Random(1))
        action = self.game.legal_actions(state, 0)[0]
        self.assertEqual(
            len(self.game.action_features(state, 0, action)),
            len(self.game.feature_names),
        )
        self.assertEqual(len(self.game.observe(state, 0)), self.game.obs_size)


if __name__ == "__main__":
    unittest.main()
