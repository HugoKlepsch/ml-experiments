"""Kuhn poker: two players, three cards, one betting round.

This is the smallest game that is genuinely a *poker* game, and it is here to
exercise the parts of the interface that Connect Four does not: **private
information** and **chance**. Rules, in full:

  * Deck of three cards, J < Q < K. Each player antes 1 and is dealt one card.
  * Player 0 acts first. Two actions exist, `pass` and `bet`:
      - `pass` when nothing is owed is a check; facing a bet it is a fold.
      - `bet` when nothing is owed puts in 1; facing a bet it is a call.
  * check-check and bet-call go to showdown; the higher card takes the pot.
    A fold gives the pot to the other player.

Small enough to solve on paper, which is the reason to have it: the game-theory
optimal value to player 0 is **-1/18 chips per hand**, about -0.056. Any agent
here can be scored against a number that is known exactly rather than against
another agent of unknown strength. Note also that the optimal strategy is
*mixed* — it bluffs the jack at a specific frequency — so a deterministic
`WeightedAgent` cannot reach it however its weights are set. `temperature` on
that agent is what buys back the randomisation.

Hidden information is enforced by `view`, which replaces the opponent's card
with None before an agent ever sees the state. Everything an agent can reach
works on a redacted state; only the runner, holding the true state, can call
`scores`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.game import TurnBasedGame

JACK, QUEEN, KING = 0, 1, 2
CARD_NAMES = ("J", "Q", "K")
DECK = (JACK, QUEEN, KING)

PASS, BET = 0, 1
ANTE = 1
BET_SIZE = 1


@dataclass(frozen=True)
class KuhnState:
    cards: tuple           # per player: card index, or None if not visible
    history: tuple         # actions taken so far, in order
    to_move: int
    # Filled in once the hand is over: net chips for each player, and whether
    # the hand was decided at showdown rather than by a fold.
    payoffs: tuple | None
    showdown: bool


def _resolved(history):
    """Whether `history` ends the hand, and if so how.

    Returns (over, folded_by) where `folded_by` is the seat that folded, or
    None when the hand reaches showdown.
    """
    if history == (PASS, PASS):
        return True, None                 # check-check
    if history == (PASS, BET, PASS):
        return True, 0                    # player 0 folds to the raise
    if history == (PASS, BET, BET):
        return True, None                 # player 0 calls
    if history == (BET, PASS):
        return True, 1                    # player 1 folds
    if history == (BET, BET):
        return True, None                 # player 1 calls
    return False, None


def _payoffs(cards, history):
    """Net chips for each player. Zero-sum, so the two always cancel."""
    over, folded = _resolved(history)
    if not over:
        return None
    # Every `bet` in the history is one more chip in from the player who made
    # it, on top of the ante both posted.
    pot_in = [ANTE, ANTE]
    for i, action in enumerate(history):
        if action == BET:
            pot_in[i % 2] += BET_SIZE
    if folded is not None:
        winner = 1 - folded
    else:
        winner = 0 if cards[0] > cards[1] else 1
    loser = 1 - winner
    # The winner collects what the loser put in; the loser is out that much.
    return tuple(
        float(pot_in[loser]) if i == winner else -float(pot_in[loser])
        for i in range(2)
    )


class KuhnGame(TurnBasedGame):
    name = "kuhn"
    num_players = 2
    action_names = ("pass", "bet")
    feature_names = (
        "strength",       # our card, scaled to 0/0.5/1
        "aggression",     # 1 for bet or call, 0 for check or fold
        "strong_bet",     # betting or calling with a good card
        "weak_bet",       # betting or calling with a bad card -- a bluff or a crying call
        "facing_bet",     # a bet is owed, so `pass` is a fold and forfeits the pot
        "pot_odds",       # chips we would win over chips we would risk
        "first_to_act",   # position: acting first is a real disadvantage here
    )
    # Card one-hot (3, all zero if we somehow cannot see it) + one-hot over the
    # four reachable decision points + position + chips already committed.
    obs_size = 3 + 4 + 1 + 1

    # Decision points a player can face, in the order the observation encodes.
    _NODES = ((), (PASS,), (BET,), (PASS, BET))

    # --- dynamics ---------------------------------------------------------

    def reset(self, rng):
        cards = rng.sample(DECK, 2)
        return KuhnState(
            cards=tuple(cards), history=(), to_move=0, payoffs=None, showdown=False
        )

    def current_player(self, state):
        return state.to_move

    def moves(self, state):
        return [PASS, BET]

    def play(self, state, player, action, rng):
        history = state.history + (action,)
        over, folded = _resolved(history)
        return KuhnState(
            cards=state.cards,
            history=history,
            to_move=1 - player,
            payoffs=_payoffs(state.cards, history) if over else None,
            showdown=over and folded is None,
        )

    def is_terminal(self, state):
        return state.payoffs is not None

    def scores(self, state):
        """Net chips won. Zero until the hand resolves, then equal and opposite.

        Only ever called on the true state — a view has a missing card and
        could not settle a showdown.
        """
        return state.payoffs if state.payoffs is not None else (0.0, 0.0)

    # --- hidden information -----------------------------------------------

    def view(self, state, player):
        """The information set `player` is in: own card, plus public history."""
        return replace(
            state,
            cards=tuple(c if i == player else None for i, c in enumerate(state.cards)),
        )

    # --- features ---------------------------------------------------------

    def _committed(self, history):
        """Chips each player has put in so far."""
        pot_in = [ANTE, ANTE]
        for i, action in enumerate(history):
            if action == BET:
                pot_in[i % 2] += BET_SIZE
        return pot_in

    def action_features(self, state, player, action):
        card = state.cards[player]
        strength = card / 2.0
        facing = 1.0 if state.history and state.history[-1] == BET else 0.0
        aggressive = 1.0 if action == BET else 0.0

        pot_in = self._committed(state.history)
        pot = sum(pot_in)
        risk = BET_SIZE if action == BET else 0.0
        # What we stand to win against what the action costs us. With nothing
        # owed and no bet made this is the pot for free, hence the guard.
        pot_odds = pot / (risk + pot_in[player]) if (risk + pot_in[player]) else 0.0

        return (
            strength,
            aggressive,
            aggressive * strength,
            aggressive * (1.0 - strength),
            facing,
            pot_odds / 3.0,
            1.0 if player == 0 else 0.0,
        )

    def observe(self, state, player, encoding=None):
        card = state.cards[player]
        card_hot = [0.0, 0.0, 0.0]
        if card is not None:
            card_hot[card] = 1.0
        node_hot = [1.0 if state.history == n else 0.0 for n in self._NODES]
        pot_in = self._committed(state.history)
        return card_hot + node_hot + [
            1.0 if player == 0 else 0.0,
            pot_in[player] / (ANTE + BET_SIZE),
        ]

    def render(self, state):
        """Spectator view, cards face up.

        `private` marks which of them a player could not see while deciding, so
        the viewer can show a card that was concealed at the time differently
        from one that went to showdown.
        """
        return {
            "kind": "kuhn",
            "cards": [CARD_NAMES[c] if c is not None else "?" for c in state.cards],
            "private": not state.showdown,
            "history": [self.action_names[a] for a in state.history],
            "pot": sum(self._committed(state.history)),
            "committed": self._committed(state.history),
            "showdown": state.showdown,
            "payoffs": list(state.payoffs) if state.payoffs else None,
        }
