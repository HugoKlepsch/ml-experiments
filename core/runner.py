"""Episode execution and recording."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.game import NO_ACTION, SIMULTANEOUS


@dataclass
class Episode:
    game: str
    seed: int
    agents: list[str]
    scores: list[float]
    winners: list[int]
    steps: int
    frames: list[dict] = field(default_factory=list)
    # Final position, always kept. Cheap, and analysis usually wants the
    # outcome (max tile reached, snake lengths) without paying for full frames.
    final: dict = field(default_factory=dict)

    def summary(self) -> str:
        parts = ", ".join(
            f"{name}={score:.0f}" for name, score in zip(self.agents, self.scores)
        )
        return f"{self.game} seed={self.seed} steps={self.steps} [{parts}]"


def poll(game, agents, state, agent_rngs) -> list[int]:
    """Ask whoever is to move for their action. Returns one entry per player.

    Turn-based games only consult the active seat; everyone else passes. Agents
    are handed `game.view(state, i)` rather than the state, which is what stops
    a poker agent from reading its opponent's card. For perfect-information
    games the view *is* the state and this costs nothing.
    """
    turn = game.current_player(state)
    movers = range(game.num_players) if turn == SIMULTANEOUS else (turn,)

    actions = [NO_ACTION] * game.num_players
    for i in movers:
        if game.legal_actions(state, i):
            actions[i] = agents[i].act(game, game.view(state, i), i, agent_rngs[i])
    return actions


def play_episode(game, agents, seed, record=False, max_steps=100_000) -> Episode:
    """Play one episode. `agents[i]` controls player i.

    The seed drives the game's randomness *and* each agent's own rng, so the
    same (agents, seed) pair always reproduces the same episode. Common random
    numbers in the arena and the GA depend on that.
    """
    if len(agents) != game.num_players:
        raise ValueError(
            f"{game.name} needs {game.num_players} agents, got {len(agents)}"
        )

    rng = random.Random(seed)
    # Separate streams per agent so that changing one agent does not shift the
    # dice for the others.
    agent_rngs = [random.Random(seed * 7919 + i) for i in range(len(agents))]
    for agent in agents:
        agent.reset()

    state = game.reset(rng)
    frames = []
    if record:
        frames.append(_frame(game, state, None))

    steps = 0
    while steps < max_steps and not game.is_terminal(state):
        actions = poll(game, agents, state, agent_rngs)
        state = game.step(state, tuple(actions), rng)
        steps += 1
        if record:
            frames.append(_frame(game, state, actions))

    return Episode(
        game=game.name,
        seed=seed,
        agents=[a.name for a in agents],
        scores=list(game.scores(state)),
        winners=game.winners(state),
        steps=steps,
        frames=frames,
        final=_frame(game, state, None),
    )


def _frame(game, state, actions):
    frame = game.render(state)
    frame["scores"] = list(game.scores(state))
    frame["alive"] = [game.alive(state, i) for i in range(game.num_players)]
    turn = None if game.is_terminal(state) else game.current_player(state)
    # `null` covers both "game over" and "everybody moves at once".
    frame["to_move"] = None if turn == SIMULTANEOUS else turn
    if actions is not None:
        frame["actions"] = [
            game.action_names[a] if a != NO_ACTION else None for a in actions
        ]
    return frame
