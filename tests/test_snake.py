"""Rules tests for multiplayer Snake, concentrating on collision resolution."""

import random
import unittest

from games.snake import DOWN, LEFT, RIGHT, UP, SnakeGame, SnakeState


def state(bodies, directions, food=(), alive=None, hunger=None, eaten=None):
    n = len(bodies)
    return SnakeState(
        bodies=tuple(tuple(b) for b in bodies),
        alive=tuple(alive if alive is not None else [True] * n),
        directions=tuple(directions),
        food=frozenset(food),
        eaten=tuple(eaten if eaten is not None else [0] * n),
        ticks=(0,) * n,
        hunger=tuple(hunger if hunger is not None else [0] * n),
        step=0,
    )


class TestMovement(unittest.TestCase):
    def setUp(self):
        self.game = SnakeGame(width=8, height=8, num_players=1)
        self.rng = random.Random(0)

    def test_head_advances_and_tail_follows(self):
        s = state([[(3, 3), (2, 3), (1, 3)]], [RIGHT])
        after = self.game.step(s, (RIGHT,), self.rng)
        self.assertEqual(after.bodies[0], ((4, 3), (3, 3), (2, 3)))

    def test_eating_grows_without_dropping_the_tail(self):
        s = state([[(3, 3), (2, 3), (1, 3)]], [RIGHT], food=[(4, 3)])
        after = self.game.step(s, (RIGHT,), self.rng)
        self.assertEqual(after.bodies[0], ((4, 3), (3, 3), (2, 3), (1, 3)))
        self.assertEqual(after.eaten[0], 1)
        self.assertNotIn((4, 3), after.food)

    def test_walking_off_the_board_kills(self):
        s = state([[(0, 3), (1, 3), (2, 3)]], [LEFT])
        self.assertFalse(self.game.step(s, (LEFT,), self.rng).alive[0])

    def test_reversing_into_your_own_neck_is_not_offered(self):
        s = state([[(3, 3), (2, 3), (1, 3)]], [RIGHT])
        self.assertNotIn(LEFT, self.game.legal_actions(s, 0))
        self.assertEqual(sorted(self.game.legal_actions(s, 0)), [UP, DOWN, RIGHT])

    def test_running_into_your_own_body_kills(self):
        # A six-segment coil. The head at (3,3) came from (3,4), so it is
        # travelling UP, and stepping LEFT re-enters the midsection at (2,3) --
        # not the neck and not the tail, so no other rule can excuse it.
        body = [(3, 3), (3, 4), (2, 4), (2, 3), (2, 2), (3, 2)]
        s = state([body], [UP])
        self.assertFalse(self.game.step(s, (LEFT,), self.rng).alive[0])

    def test_chasing_your_own_vacating_tail_survives(self):
        # Head (2,2) came from (2,3) so it is travelling UP; stepping RIGHT
        # enters (3,2), the tail cell, which empties on this same tick.
        s = state([[(2, 2), (2, 3), (3, 3), (3, 2)]], [UP])
        self.assertTrue(self.game.step(s, (RIGHT,), self.rng).alive[0])

    def test_starvation_kills(self):
        game = SnakeGame(width=8, height=8, num_players=1, starve_limit=1)
        s = state([[(3, 3), (2, 3), (1, 3)]], [RIGHT])
        self.assertFalse(game.step(s, (RIGHT,), self.rng).alive[0])


class TestMultiplayerCollisions(unittest.TestCase):
    def setUp(self):
        self.game = SnakeGame(width=9, height=9, num_players=2)
        self.rng = random.Random(0)

    def test_head_on_collision_kills_both(self):
        s = state(
            [[(3, 4), (2, 4), (1, 4)], [(5, 4), (6, 4), (7, 4)]], [RIGHT, LEFT]
        )
        after = self.game.step(s, (RIGHT, LEFT), self.rng)
        self.assertEqual(after.alive, (False, False))

    def test_head_on_kills_both_regardless_of_length(self):
        s = state(
            [[(3, 4), (2, 4)], [(5, 4), (6, 4), (7, 4), (8, 4)]], [RIGHT, LEFT]
        )
        self.assertEqual(self.game.step(s, (RIGHT, LEFT), self.rng).alive, (False, False))

    def test_running_into_a_rival_body_kills_only_the_mover(self):
        # Player 1 runs vertically and steps DOWN to (5,7), so (5,4) is still
        # occupied midsection rather than a vacating tail. Player 0 steps into it.
        s = state(
            [[(4, 4), (3, 4), (2, 4)], [(5, 6), (5, 5), (5, 4), (5, 3)]],
            [RIGHT, DOWN],
        )
        after = self.game.step(s, (RIGHT, DOWN), self.rng)
        self.assertFalse(after.alive[0])
        self.assertTrue(after.alive[1])

    def test_game_ends_when_one_snake_remains(self):
        s = state(
            [[(0, 4), (1, 4), (2, 4)], [(5, 5), (5, 4), (5, 3)]], [LEFT, DOWN]
        )
        after = self.game.step(s, (LEFT, DOWN), self.rng)
        self.assertEqual(after.alive, (False, True))  # player 0 walked off the edge
        self.assertTrue(self.game.is_terminal(after))

    def test_dead_players_have_no_legal_actions(self):
        s = state([[(3, 4)], [(5, 4)]], [RIGHT, LEFT], alive=[False, True])
        self.assertEqual(self.game.legal_actions(s, 0), [])
        self.assertTrue(self.game.legal_actions(s, 1))


class TestScoringAndFeatures(unittest.TestCase):
    def setUp(self):
        self.game = SnakeGame(width=9, height=9, num_players=2)

    def test_food_dominates_survival_in_the_score(self):
        s = state([[(3, 4)], [(5, 4)]], [RIGHT, LEFT], eaten=[1, 0])
        scores = self.game.scores(s)
        self.assertGreater(scores[0], scores[1])

    def test_fatal_moves_are_flagged_by_the_features(self):
        s = state([[(0, 4), (1, 4), (2, 4)], [(5, 4), (6, 4), (7, 4)]], [LEFT, RIGHT])
        features = dict(zip(self.game.feature_names,
                            self.game.action_features(s, 0, LEFT)))
        self.assertEqual(features["dies"], 1.0)
        safe = dict(zip(self.game.feature_names,
                        self.game.action_features(s, 0, UP)))
        self.assertEqual(safe["dies"], 0.0)

    def test_moving_toward_food_is_scored_as_closer(self):
        s = state([[(3, 4), (2, 4)], [(7, 7), (7, 6)]], [RIGHT, UP], food=[(5, 4)])
        towards = self.game.action_features(s, 0, RIGHT)
        index = self.game.feature_names.index("food_closer")
        self.assertGreater(towards[index], 0)

    def test_widths_match_declarations(self):
        s = self.game.reset(random.Random(3))
        self.assertEqual(
            len(self.game.action_features(s, 0, UP)), len(self.game.feature_names)
        )
        self.assertEqual(len(self.game.observe(s, 0)), self.game.obs_size)

    def test_reset_places_distinct_snakes_and_food(self):
        s = self.game.reset(random.Random(5))
        cells = [c for body in s.bodies for c in body]
        self.assertEqual(len(cells), len(set(cells)))
        self.assertFalse(set(cells) & set(s.food))


if __name__ == "__main__":
    unittest.main()
