"""The Game interface every game must implement.

Design notes worth knowing before adding a game:

* Moves are **simultaneous**. Every step takes one action per player. A
  single-player game like 2048 is just the N=1 case, and a strictly turn-based
  game would mark all but the active player as having no legal actions. This is
  the only model that covers 2048 and N-player Snake without special cases.

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
        """JSON-serialisable snapshot for the web visualiser."""

    def alive(self, state, player) -> bool:
        """Whether the player is still participating. Default: while not terminal."""
        return not self.is_terminal(state)

    def winners(self, state) -> list[int]:
        """Indices of the highest-scoring players. Ties return several."""
        scores = self.scores(state)
        best = max(scores)
        return [i for i, s in enumerate(scores) if s == best]

    def reward(self, state, next_state, player) -> float:
        """Per-step reward for RL. Default: change in this player's score."""
        return self.scores(next_state)[player] - self.scores(state)[player]
