"""Agent interface plus the two model-free baselines.

An agent sees the game and the state and returns one legal action. Agents must
be picklable so that arena and GA evaluation can run across processes.
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
