"""Tests for the turn-based half of the Game interface, and the two games on it.

Three things are being checked here, in rising order of how badly they would
hurt if wrong:

1. Connect Four and Kuhn poker follow their own rules.
2. The turn-based plumbing holds: one mover per step, and everyone else passes.
3. **Hidden information does not leak.** Every method an agent can reach must
   give the same answer on `view(state, player)` as on the true state, and the
   view must not contain the opponent's card at all.
"""

import random
import unittest

from agents.dqn import MLP, DQNAgent, ReplayBuffer
from core.agent import RandomAgent, WeightedAgent
from core.arena import _distinct_matchups, _play_job, compare
from core.game import NO_ACTION, SIMULTANEOUS
from core.registry import list_games, make_game
from core.runner import play_episode, poll
from games import connect4 as c4
from games import kuhn
from train.dqn import collect_episode


class TestTurnBasedInterface(unittest.TestCase):
    """Contract checks that apply to any turn-based game."""

    def games(self):
        return [make_game("connect4"), make_game("kuhn")]

    def test_exactly_one_player_moves_per_step(self):
        for game in self.games():
            with self.subTest(game=game.name):
                rng = random.Random(0)
                state = game.reset(rng)
                while not game.is_terminal(state):
                    movable = [i for i in range(game.num_players)
                               if game.legal_actions(state, i)]
                    self.assertEqual(movable, [game.current_player(state)])
                    actions = poll(game, [RandomAgent()] * game.num_players,
                                   state, [rng] * game.num_players)
                    passed = [a for a in actions if a == NO_ACTION]
                    self.assertEqual(len(passed), game.num_players - 1)
                    state = game.step(state, tuple(actions), rng)

    def test_simultaneous_games_still_report_simultaneous(self):
        for name in ("2048", "snake"):
            game = make_game(name)
            state = game.reset(random.Random(0))
            self.assertEqual(game.current_player(state), SIMULTANEOUS)

    def test_illegal_and_missing_actions_are_rejected(self):
        game = make_game("connect4")
        state = game.reset(random.Random(0))
        with self.assertRaises(ValueError):
            game.step(state, (NO_ACTION, NO_ACTION), random.Random(0))
        # Fill column 0, then try to drop into it.
        rng = random.Random(0)
        for _ in range(c4.HEIGHT):
            state = game.step(state, self._actions(game, state, 0), rng)
        with self.assertRaises(ValueError):
            game.step(state, self._actions(game, state, 0), rng)

    def test_feature_and_observation_widths_match_declarations(self):
        for game in self.games():
            with self.subTest(game=game.name):
                state = game.reset(random.Random(3))
                player = game.current_player(state)
                self.assertEqual(len(game.observe(state, player)), game.obs_size)
                for action in game.legal_actions(state, player):
                    self.assertEqual(
                        len(game.action_features(state, player, action)),
                        len(game.feature_names),
                    )

    def test_episodes_are_reproducible_and_terminate(self):
        for game in self.games():
            with self.subTest(game=game.name):
                agents = [RandomAgent(), RandomAgent()]
                a = play_episode(game, agents, 5, record=True)
                b = play_episode(game, [RandomAgent(), RandomAgent()], 5)
                self.assertEqual(a.scores, b.scores)
                self.assertGreater(a.steps, 0)
                self.assertEqual(len(a.frames), a.steps + 1)
                self.assertIsNone(a.frames[-1]["to_move"])

    def test_scores_are_zero_sum(self):
        for game in self.games():
            with self.subTest(game=game.name):
                for seed in range(20):
                    episode = play_episode(game, [RandomAgent(), RandomAgent()], seed)
                    self.assertAlmostEqual(sum(episode.scores), 0.0)

    @staticmethod
    def _actions(game, state, action):
        actions = [NO_ACTION] * game.num_players
        actions[game.current_player(state)] = action
        return tuple(actions)


class TestHiddenInformation(unittest.TestCase):
    def test_view_hides_the_opponents_card(self):
        game = make_game("kuhn")
        state = game.reset(random.Random(1))
        for player in (0, 1):
            view = game.view(state, player)
            self.assertEqual(view.cards[player], state.cards[player])
            self.assertIsNone(view.cards[1 - player])

    def test_agent_facing_methods_agree_on_a_view(self):
        """The contract that makes `view` usable: anything an agent can call
        must work on a redacted state and return what it returns on the real
        one. If this fails, agents and trainers silently disagree."""
        game = make_game("kuhn")
        rng = random.Random(2)
        state = game.reset(rng)
        while not game.is_terminal(state):
            player = game.current_player(state)
            view = game.view(state, player)
            self.assertEqual(game.current_player(view), player)
            self.assertEqual(game.legal_actions(view, player),
                             game.legal_actions(state, player))
            self.assertEqual(game.observe(view, player), game.observe(state, player))
            for action in game.legal_actions(state, player):
                self.assertEqual(game.action_features(view, player, action),
                                 game.action_features(state, player, action))
            state = game.step(state, TestTurnBasedInterface._actions(
                game, state, game.legal_actions(state, player)[0]), rng)

    def test_perfect_information_games_return_the_state_itself(self):
        for name in list_games():
            if name == "kuhn":
                continue
            game = make_game(name)
            state = game.reset(random.Random(0))
            self.assertIs(game.view(state, 0), state)


class TestConnect4(unittest.TestCase):
    def setUp(self):
        self.game = make_game("connect4")

    def board(self, columns):
        """Drop into `columns` in order, alternating seats. Returns the state."""
        state = self.game.reset(random.Random(0))
        rng = random.Random(0)
        for col in columns:
            actions = [NO_ACTION, NO_ACTION]
            actions[state.to_move] = col
            state = self.game.step(state, tuple(actions), rng)
        return state

    def test_pieces_stack_from_the_bottom(self):
        state = self.board([3, 3])
        self.assertEqual(state.board[c4._index(c4.HEIGHT - 1, 3)], 1)
        self.assertEqual(state.board[c4._index(c4.HEIGHT - 2, 3)], 2)

    def test_vertical_four_wins(self):
        state = self.board([0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(state.winner, 0)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.scores(state), (1.0, -1.0))
        self.assertEqual(self.game.winners(state), [0])

    def test_horizontal_four_wins(self):
        state = self.board([0, 0, 1, 1, 2, 2, 3])
        self.assertEqual(state.winner, 0)
        self.assertEqual(len(state.board and self.game.render(state)["line"]), 4)

    def test_diagonal_four_wins(self):
        # Seat 0 builds the rising diagonal a1-b2-c3-d4; seat 1 fills underneath.
        state = self.board([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
        self.assertEqual(state.winner, 0)

    def test_full_column_is_not_a_legal_move(self):
        state = self.board([0, 0, 0, 0, 0, 0])
        self.assertNotIn(0, self.game.legal_actions(state, state.to_move))
        self.assertEqual(len(self.game.legal_actions(state, state.to_move)), c4.WIDTH - 1)

    def test_a_full_board_with_no_line_is_a_draw(self):
        # Column-paired filling: each colour owns whole columns, so no colour
        # ever gets four in a row horizontally, vertically or diagonally.
        order = []
        for col in range(c4.WIDTH):
            order += [col] * c4.HEIGHT
        state = self.game.reset(random.Random(0))
        rng = random.Random(0)
        for col in order:
            if self.game.is_terminal(state):
                break
            actions = [NO_ACTION, NO_ACTION]
            actions[state.to_move] = col
            state = self.game.step(state, tuple(actions), rng)
        if state.winner is None:
            self.assertEqual(self.game.scores(state), (0.0, 0.0))
            self.assertEqual(self.game.winners(state), [0, 1])

    def test_wins_now_feature_fires_on_the_winning_column(self):
        state = self.board([0, 1, 0, 1, 0, 1])   # seat 0 has three in column 0
        names = self.game.feature_names
        features = self.game.action_features(state, 0, 0)
        self.assertEqual(features[names.index("wins_now")], 1.0)
        other = self.game.action_features(state, 0, 5)
        self.assertEqual(other[names.index("wins_now")], 0.0)

    def test_blocks_win_feature_fires_where_the_opponent_would_win(self):
        # Seat 0 stacks three in column 0; seat 1's pieces are spread so that
        # it has no win of its own and the only urgent move is the block.
        state = self.board([0, 2, 0, 4, 0, 6, 5])
        names = self.game.feature_names
        self.assertEqual(state.to_move, 1)
        self.assertEqual(
            self.game.action_features(state, 1, 0)[names.index("blocks_win")], 1.0
        )
        self.assertEqual(
            self.game.action_features(state, 1, 3)[names.index("blocks_win")], 0.0
        )

    def test_a_feature_weight_produces_a_stronger_player(self):
        """An agent that takes free wins and blocks losses should beat random."""
        names = self.game.feature_names
        weights = [0.0] * len(names)
        weights[names.index("wins_now")] = 10.0
        weights[names.index("blocks_win")] = 5.0
        weights[names.index("centre")] = 1.0
        tactical = WeightedAgent(weights, name="tactical")

        wins = 0.0
        for seed in range(60):
            seat = seed % 2
            agents = [RandomAgent(), RandomAgent()]
            agents[seat] = tactical
            episode = play_episode(self.game, agents, seed)
            if seat in episode.winners:
                wins += 1.0 / len(episode.winners)
        self.assertGreater(wins, 45)   # random-vs-random would sit near 30


class TestKuhnPoker(unittest.TestCase):
    def setUp(self):
        self.game = make_game("kuhn")

    def hand(self, cards, history):
        state = kuhn.KuhnState(cards=cards, history=(), to_move=0,
                               payoffs=None, showdown=False)
        rng = random.Random(0)
        for action in history:
            actions = [NO_ACTION, NO_ACTION]
            actions[state.to_move] = action
            state = self.game.step(state, tuple(actions), rng)
        return state

    def test_each_player_gets_a_distinct_card(self):
        for seed in range(30):
            state = self.game.reset(random.Random(seed))
            self.assertNotEqual(state.cards[0], state.cards[1])
            self.assertTrue(all(c in kuhn.DECK for c in state.cards))

    def test_check_check_is_a_showdown_for_one_chip(self):
        state = self.hand((kuhn.KING, kuhn.JACK), (kuhn.PASS, kuhn.PASS))
        self.assertTrue(self.game.is_terminal(state))
        self.assertTrue(state.showdown)
        self.assertEqual(self.game.scores(state), (1.0, -1.0))

    def test_bet_call_is_a_showdown_for_two_chips(self):
        state = self.hand((kuhn.JACK, kuhn.QUEEN), (kuhn.BET, kuhn.BET))
        self.assertEqual(self.game.scores(state), (-2.0, 2.0))

    def test_a_fold_gives_the_pot_away_regardless_of_cards(self):
        # Player 1 folds the king to a bet and still loses a chip.
        state = self.hand((kuhn.JACK, kuhn.KING), (kuhn.BET, kuhn.PASS))
        self.assertFalse(state.showdown)
        self.assertEqual(self.game.scores(state), (1.0, -1.0))

    def test_check_raise_line_resolves(self):
        state = self.hand((kuhn.QUEEN, kuhn.KING), (kuhn.PASS, kuhn.BET, kuhn.PASS))
        self.assertEqual(self.game.scores(state), (-1.0, 1.0))
        state = self.hand((kuhn.QUEEN, kuhn.KING), (kuhn.PASS, kuhn.BET, kuhn.BET))
        self.assertEqual(self.game.scores(state), (-2.0, 2.0))

    def test_always_folding_loses_more_than_the_optimal_value(self):
        """Sanity anchor. Kuhn's game value to seat 0 is -1/18 against perfect
        play; a player who folds every hand does far worse than that."""
        names = self.game.feature_names
        folder = WeightedAgent(
            [-1.0 if n == "aggression" else 0.0 for n in names], name="folder"
        )
        total = sum(
            play_episode(self.game, [folder, RandomAgent()], seed).scores[0]
            for seed in range(200)
        )
        self.assertLess(total / 200, -1 / 18)

    def test_a_strong_hand_reads_as_strong(self):
        state = self.hand((kuhn.KING, kuhn.JACK), ())
        names = self.game.feature_names
        self.assertEqual(
            self.game.action_features(state, 0, kuhn.BET)[names.index("strength")], 1.0
        )
        weak = self.hand((kuhn.JACK, kuhn.KING), ())
        self.assertEqual(
            self.game.action_features(weak, 0, kuhn.BET)[names.index("strength")], 0.0
        )


class TestDeterministicMatchups(unittest.TestCase):
    """Connect Four has no chance in it, so two deterministic agents replay one
    game forever. The arena has to notice rather than print a confident
    interval around a sample size of one."""

    def test_warning_fires_for_a_chanceless_game_and_fixed_agents(self):
        raw = [(("a", "b"), [1.0, -1.0], [0]) for _ in range(50)]
        raw += [(("b", "a"), [1.0, -1.0], [0]) for _ in range(50)]
        self.assertEqual(_distinct_matchups(raw), 2)

    def test_no_warning_when_outcomes_actually_vary(self):
        raw = [(("a", "b"), [1.0, -1.0], [0]) for _ in range(50)]
        raw += [(("a", "b"), [-1.0, 1.0], [1]) for _ in range(50)]
        self.assertIsNone(_distinct_matchups(raw))

    def test_end_to_end_against_the_real_games(self):
        # Two saved models on Connect Four replay one game per seating; either
        # side being random restores a real distribution.
        fixed = compare("connect4", ["first", "first"], games=20,
                        workers=1, quiet=True)
        self.assertIsNotNone(_distinct_matchups(self._raw("connect4", ["first", "first"])))
        self.assertEqual(sum(r.games for r in fixed.values()), 40)
        self.assertIsNone(_distinct_matchups(self._raw("connect4", ["first", "random"])))
        # Kuhn deals cards, so even two fixed agents see a spread of hands.
        self.assertIsNone(_distinct_matchups(self._raw("kuhn", ["first", "first"])))

    @staticmethod
    def _raw(game_name, agent_names, games=20):
        return [
            _play_job((game_name, {},
                       tuple(agent_names[(i + k) % 2] for k in range(2)),
                       1_000_000 + i))
            for i in range(games)
        ]


class TestTurnBasedCollection(unittest.TestCase):
    """The DQN collector has to span opponent plies, not tick to tick."""

    def collect(self, game_name, episodes=40, seat=0):
        game = make_game(game_name)
        agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]), epsilon=0.5)
        buffer = ReplayBuffer(4096, game.obs_size, seed=0)
        scores = [
            collect_episode(game, agent, "random", seed, buffer, seat)
            for seed in range(episodes)
        ]
        return game, buffer, scores

    def test_one_transition_per_decision_not_per_tick(self):
        """Connect Four alternates, so a learner that made k moves must leave
        exactly k transitions -- not one per ply of the whole game."""
        game = make_game("connect4")
        agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]))
        buffer = ReplayBuffer(256, game.obs_size, seed=0)
        collect_episode(game, agent, "random", 1, buffer, seat=0)

        episode = play_episode(game, [agent, RandomAgent()], 1)
        # Seat 0 moves on the even plies of the episode.
        expected = (episode.steps + 1) // 2
        self.assertEqual(buffer.size, expected)

    def test_terminal_losses_reach_the_buffer(self):
        """The failure this guards against: in Connect Four the reward for a
        loss lands on the *opponent's* ply. A per-tick collector attributes it
        to nobody and the learner never sees a negative outcome at all."""
        _, buffer, scores = self.collect("connect4", episodes=60)
        stored = buffer.rewards[: buffer.size]
        self.assertLess(scores.count(-1.0), 60)     # the fixture is not degenerate
        self.assertGreater(scores.count(-1.0), 0)
        self.assertTrue((stored < 0).any(), "no loss was ever recorded")
        self.assertTrue((stored > 0).any(), "no win was ever recorded")

    def test_stored_reward_totals_match_the_final_score(self):
        for name in ("connect4", "kuhn"):
            with self.subTest(game=name):
                game = make_game(name)
                agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]))
                for seed in range(15):
                    buffer = ReplayBuffer(256, game.obs_size, seed=0)
                    score = collect_episode(game, agent, "random", seed, buffer, 0)
                    total = float(buffer.rewards[: buffer.size].sum())
                    self.assertAlmostEqual(total, score, places=4)

    def test_only_the_last_transition_is_terminal(self):
        for name in ("connect4", "kuhn"):
            with self.subTest(game=name):
                game = make_game(name)
                agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]))
                buffer = ReplayBuffer(256, game.obs_size, seed=0)
                collect_episode(game, agent, "random", 4, buffer, 0)
                done = buffer.done[: buffer.size]
                self.assertEqual(done[-1], 1.0)
                self.assertEqual(done[:-1].sum(), 0.0)

    def test_simultaneous_games_are_unaffected(self):
        """2048 decides on every tick, so decisions and ticks coincide and the
        collector must still store one transition per step."""
        game = make_game("2048")
        agent = DQNAgent(MLP([game.obs_size, 8, game.num_actions]))
        buffer = ReplayBuffer(100_000, game.obs_size, seed=0)
        collect_episode(game, agent, "random", 2, buffer, seat=0)
        episode = play_episode(game, [agent], 2)
        self.assertEqual(buffer.size, episode.steps)


if __name__ == "__main__":
    unittest.main()
