"""Agent interface plus the two model-free baselines.

An agent sees the game and the state and returns one legal action. Agents must
be picklable so that arena and GA evaluation can run across processes.

Agents are also asked for **several decisions at once** by the vectorised
collector in `train/dqn.py`, through `act_batch`. The default implementation
just loops, so nothing has to opt in; a neural agent overrides it to run one
forward pass over the whole batch instead of one per position, which is the
entire point of collecting from many environments at a time.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from core.game import NO_ACTION


class Agent(ABC):
    name: str

    @abstractmethod
    def act(self, game, state, player: int, rng: random.Random) -> int:
        ...

    def act_batch(self, game, states, players, rngs, observations=None) -> list[int]:
        """One action per position, for positions from *different* episodes.

        Each entry has its own `rng`, so an agent must draw from `rngs[k]` for
        row k and nothing else -- that is what keeps a batched run identical to
        the same episodes played one at a time.

        `observations` is an optional list the caller has already built (it
        needs them for the replay buffer anyway). An agent that consumes
        observations should use them rather than calling `observe` again;
        building one is comparable in cost to a small forward pass.

        The default loops. Override only to batch something expensive.
        """
        return [
            self.act(game, state, player, rng)
            for state, player, rng in zip(states, players, rngs)
        ]

    def reset(self) -> None:
        """Called once at the start of each episode."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class RandomAgent(Agent):
    """Uniform over legal actions. The floor every model must beat."""

    def __init__(self, name="random"):
        self.name = name

    def act(self, game, state, player, rng):
        options = game.legal_actions(state, player)
        return rng.choice(options) if options else NO_ACTION


class FirstActionAgent(Agent):
    """Always takes the lowest-numbered legal action. Deterministic control."""

    def __init__(self, name="first"):
        self.name = name

    def act(self, game, state, player, rng):
        options = game.legal_actions(state, player)
        return options[0] if options else NO_ACTION


class WeightedAgent(Agent):
    """Greedy over a linear score of each action's features.

    This is the agent the GA trains: evolution only sets `weights`. Because the
    features come from the game, the same class plays every game.

    `temperature` > 0 samples softmax-style instead of taking the argmax, which
    is useful for generating varied self-play data.
    """

    def __init__(self, weights, name="weighted", temperature=0.0):
        self.weights = tuple(weights)
        self.name = name
        self.temperature = temperature

    def score_action(self, game, state, player, action):
        features = game.action_features(state, player, action)
        return sum(w * f for w, f in zip(self.weights, features))

    def act(self, game, state, player, rng):
        options = game.legal_actions(state, player)
        if not options:
            return NO_ACTION
        values = [self.score_action(game, state, player, a) for a in options]
        if self.temperature <= 0:
            best = max(range(len(options)), key=values.__getitem__)
            return options[best]
        peak = max(values)
        weights = [math.exp((v - peak) / self.temperature) for v in values]
        return rng.choices(options, weights=weights)[0]
