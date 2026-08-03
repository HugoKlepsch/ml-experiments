"""Local web UI for watching matches between trained models.

Run:  .venv/bin/python serve.py   then open http://127.0.0.1:8000

Deliberately stdlib-only and bound to localhost. It runs whatever agents it
finds in models/, so anything you train shows up in the dropdowns on refresh.
"""

from __future__ import annotations

import argparse
import json
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.registry import list_agents, list_games, load_agent, make_game
from core.runner import play_episode

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_FRAMES = 4000


def describe_game(name):
    game = make_game(name)
    return {
        "name": name,
        "default_players": game.num_players,
        # Snake seats an arbitrary number of players; 2048 is fixed at one.
        "variable_players": name == "snake",
        "max_players": 8 if name == "snake" else 1,
        "actions": list(game.action_names),
        "features": list(game.feature_names),
    }


def run_match(payload):
    game_name = payload["game"]
    agent_names = payload["agents"]
    seed = int(payload.get("seed", 0))

    kwargs = {}
    if game_name == "snake":
        kwargs["num_players"] = len(agent_names)
        if "board" in payload:
            size = max(6, min(24, int(payload["board"])))
            kwargs["width"] = kwargs["height"] = size

    game = make_game(game_name, **kwargs)
    agents = [load_agent(game_name, n) for n in agent_names]
    episode = play_episode(game, agents, seed, record=True)

    frames = episode.frames
    stride = max(1, len(frames) // MAX_FRAMES)
    return {
        "game": episode.game,
        "seed": episode.seed,
        "agents": episode.agents,
        "scores": episode.scores,
        "winners": episode.winners,
        "steps": episode.steps,
        "truncated": stride > 1,
        "frames": frames[::stride],
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        pass  # keep the console clear for training output

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/games"):
            return self._send_json([describe_game(n) for n in list_games()])
        if self.path.startswith("/api/agents"):
            game = self.path.split("game=")[-1].split("&")[0] or "2048"
            return self._send_json(list_agents(game))
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/play"):
            return self._send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            return self._send_json(run_match(payload))
        except Exception as exc:  # surface the reason in the UI, not just a 500
            traceback.print_exc()
            return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"visualiser on http://{args.host}:{args.port}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
