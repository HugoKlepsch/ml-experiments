"""The Game interface every game must implement.

Design notes worth knowing before adding a game:

* **Two turn structures.** `current_player(state)` says who moves next. It
  returns `SIMULTANEOUS` for games where everybody moves at once (2048 with one
  player, N-player Snake), or a seat index for turn-based games where exactly
  one player moves per step (Connect-4, Kuhn poker). `step` still takes one
  action per player either way; in a turn-based game every seat but the active
  one passes `NO_ACTION`. Subclass `TurnBasedGame` and you never write that
  boilerplate — you implement `moves` and `play` for the player to move.

* **Hidden information** is expressed by `view(state, player)`, which returns
  what that player is allowed to know. The runner hands agents the *view*, not
  the state, so an agent physically cannot read an opponent's hole card. The
  contract that makes this work: every method an agent can reach —
  `legal_actions`, `action_features`, `observe` — must accept a view as well as
  a full state. Perfect-information games return the state unchanged and pay
  nothing for any of this. `tests/test_core.py` checks the contract.

* Games expose two different views of a position, because the two families of
  agent need different things:

  `action_features(state, player, action)` scores a *candidate action* and is
  what the weighted/GA agents consume. Putting it on the game means one GA
  trainer works for every game.

  `observe(state, player)` is a fixed-length vector describing the position and
  is what the DQN consumes, since a Q-network maps one state to all actions.

* States must be treated as immutable. `step` returns a new state. The runner
  keeps old states around for replay, so mutating in place corrupts history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# Passed for a player who has no legal actions this step (dead, or not to move).
NO_ACTION = -1

# Returned by `current_player` when every player acts on the same tick.
SIMULTANEOUS = -2


class Game(ABC):
    name: str
    num_players: int
    action_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    obs_size: int

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    @abstractmethod
    def reset(self, rng):
        """Return the initial state."""

    @abstractmethod
    def legal_actions(self, state, player) -> list[int]:
        """Actions `player` may take. Empty means the player cannot move."""

    @abstractmethod
    def step(self, state, actions, rng):
        """Advance one tick. `actions` has one entry per player."""

    @abstractmethod
    def is_terminal(self, state) -> bool:
        ...

    @abstractmethod
    def scores(self, state) -> tuple[float, ...]:
        """Per-player score. Higher is better. Used to rank and to declare winners."""

    @abstractmethod
    def action_features(self, state, player, action) -> tuple[float, ...]:
        """Features of taking `action`. Must match `feature_names` in length."""

    @abstractmethod
    def observe(self, state, player) -> list[float]:
        """Fixed-length observation vector of length `obs_size`."""

    @abstractmethod
    def render(self, state) -> dict:
        """JSON-serialisable snapshot for the web visualiser.

        This is the spectator view and may reveal private information: it is
        only ever built for a replay of a finished episode, never handed to an
        agent. Games with hidden information should mark what was private so
        the viewer can present it as such.
        """

    def current_player(self, state) -> int:
        """Seat to move, or `SIMULTANEOUS` if everybody moves at once."""
        return SIMULTANEOUS

    def view(self, state, player):
        """What `player` is allowed to know. Default: the whole state."""
        return state

    def alive(self, state, player) -> bool:
        """Whether the player is still participating. Default: while not terminal."""
        return not self.is_terminal(state)

    def winners(self, state) -> list[int]:
        """Indices of the highest-scoring players. Ties return several.

        A drawn two-player game therefore returns both seats, which is what the
        arena wants: it splits the win and each side is credited half.
        """
        scores = self.scores(state)
        best = max(scores)
        return [i for i, s in enumerate(scores) if s == best]

    def reward(self, state, next_state, player) -> float:
        """Per-step reward for RL. Default: change in this player's score."""
        return self.scores(next_state)[player] - self.scores(state)[player]


class TurnBasedGame(Game):
    """Exactly one player moves per step.

    Subclasses implement `current_player`, `moves` and `play`, all phrased in
    terms of the single player to move; the action-tuple plumbing the rest of
    the bench speaks is handled here. Everything downstream — both trainers,
    the arena, the viewer — works unchanged, because from the outside a
    turn-based game is just one where all but one seat has no legal action.
    """

    @abstractmethod
    def current_player(self, state) -> int:
        """Seat to move. Only called on non-terminal states."""

    @abstractmethod
    def moves(self, state) -> list[int]:
        """Actions available to the player to move."""

    @abstractmethod
    def play(self, state, player, action, rng):
        """Apply one player's action and return the new state."""

    def legal_actions(self, state, player):
        if self.is_terminal(state) or player != self.current_player(state):
            return []
        return self.moves(state)

    def step(self, state, actions, rng):
        player = self.current_player(state)
        action = actions[player]
        if action == NO_ACTION:
            raise ValueError(f"{self.name}: player {player} is to move but passed")
        if action not in self.moves(state):
            raise ValueError(f"{self.name}: illegal action {action!r} for player {player}")
        return self.play(state, player, action, rng)
