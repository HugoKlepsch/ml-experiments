"""Lookup of games and agents by name.

Trained models live in `models/<game>/<name>.json` and are discovered from disk,
so anything you train shows up in the arena and the web UI without code changes.
Each model file records the `kind` that tells us how to rebuild the agent.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _games():
    from games.g2048 import Game2048
    from games.snake import SnakeGame

    return {"2048": Game2048, "snake": SnakeGame}


def list_games() -> list[str]:
    return sorted(_games())


def make_game(name: str, **kwargs):
    games = _games()
    if name not in games:
        raise KeyError(f"unknown game {name!r}; have {sorted(games)}")
    return games[name](**kwargs)


def model_path(game: str, name: str) -> Path:
    return MODEL_DIR / game / f"{name}.json"


def save_model(game: str, name: str, payload: dict) -> Path:
    path = model_path(game, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"game": game, "name": name, **payload}
    path.write_text(json.dumps(payload, indent=2))
    return path


def list_agents(game: str) -> list[str]:
    """Built-in baselines plus every model trained for this game."""
    names = ["random", "first"]
    directory = MODEL_DIR / game
    if directory.is_dir():
        names += sorted(p.stem for p in directory.glob("*.json"))
    return names


def load_agent(game: str, name: str):
    """Build an agent by name. Baselines first, then saved models."""
    from core.agent import FirstActionAgent, RandomAgent, WeightedAgent

    if name == "random":
        return RandomAgent()
    if name == "first":
        return FirstActionAgent()

    path = model_path(game, name)
    if not path.is_file():
        raise KeyError(
            f"no agent {name!r} for {game!r}; have {list_agents(game)}"
        )
    payload = json.loads(path.read_text())
    kind = payload.get("kind", "weighted")

    if kind == "weighted":
        return WeightedAgent(payload["weights"], name=name)
    if kind == "dqn":
        from agents.dqn import DQNAgent

        return DQNAgent.from_payload(payload, name=name)
    raise ValueError(f"unknown model kind {kind!r} in {path}")
