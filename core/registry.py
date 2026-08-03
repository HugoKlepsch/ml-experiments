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


def model_path(game: str, name: str, sweep: bool = False) -> Path:
    """Where a model is written.

    Sweep output goes in a `sweeps/` subdirectory. Those are experiment
    artifacts -- regenerated wholesale on every notebook run and large for a
    wide network -- so they are kept out of version control, while curated
    models sit alongside and are committed.
    """
    base = MODEL_DIR / game
    return base / "sweeps" / f"{name}.json" if sweep else base / f"{name}.json"


def find_model(game: str, name: str) -> Path | None:
    """Locate a model by name, preferring a curated one over a swept one."""
    for candidate in (model_path(game, name), model_path(game, name, sweep=True)):
        if candidate.is_file():
            return candidate
    return None


def save_model(game: str, name: str, payload: dict, sweep: bool = False) -> Path:
    path = model_path(game, name, sweep=sweep)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"game": game, "name": name, **payload}
    path.write_text(json.dumps(payload, indent=2))
    return path


def list_agents(game: str) -> list[str]:
    """Built-in baselines, curated models, then swept ones."""
    names = ["random", "first"]
    directory = MODEL_DIR / game
    if directory.is_dir():
        names += sorted(p.stem for p in directory.glob("*.json"))
        names += sorted(p.stem for p in (directory / "sweeps").glob("*.json"))
    # A curated model shadows a swept one of the same name; list it once.
    return list(dict.fromkeys(names))


def load_agent(game: str, name: str):
    """Build an agent by name. Baselines first, then saved models."""
    from core.agent import FirstActionAgent, RandomAgent, WeightedAgent

    if name == "random":
        return RandomAgent()
    if name == "first":
        return FirstActionAgent()

    path = find_model(game, name)
    if path is None:
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
