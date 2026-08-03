"""Train the same model repeatedly while varying one parameter, then measure.

The measuring half matters more than the training half. Retraining with a
different hidden size is easy; deciding whether the result is genuinely better
is the part that goes wrong, so every number here ships with an interval and
every variant is scored on the same seeds against the same reference opponent.

Typical use, from a notebook:

    from train.sweep import sweep, round_robin
    rows = sweep("dqn", "snake", "hidden", [32, 64, 128, 256], players=2)

Each run is saved as its own model, so swept variants show up in the arena and
the match viewer alongside everything else.
"""

from __future__ import annotations

import math
import multiprocessing
import statistics
import time

from core.arena import _play_job, wilson_interval
from core.registry import make_game

TRAINERS = {"ga": "train.ga", "dqn": "train.dqn"}


def _trainer(kind):
    if kind == "ga":
        from train.ga import train_ga

        return train_ga
    if kind == "dqn":
        from train.dqn import train_dqn

        return train_dqn
    raise ValueError(f"unknown trainer {kind!r}; expected one of {sorted(TRAINERS)}")


def evaluate_model(game_name, name, *, games=400, opponent="random", players=None,
                   seed_offset=5_000_000, workers=None):
    """Score one saved model against a reference opponent.

    Multiplayer: the model is seated against `opponent` in every other seat, and
    the seat rotates so position cannot flatter it. Returns mean score with a
    standard error, plus a win rate with a Wilson interval.
    """
    kwargs = {"num_players": players} if players else {}
    game = make_game(game_name, **kwargs)
    seats = game.num_players

    jobs = []
    for i in range(games):
        seed = seed_offset + i
        if seats == 1:
            jobs.append((game_name, kwargs, (name,), seed))
        else:
            seat = i % seats
            seating = tuple(name if k == seat else opponent for k in range(seats))
            jobs.append((game_name, kwargs, seating, seed))

    workers = workers or multiprocessing.cpu_count()
    if workers == 1:
        raw = [_play_job(j) for j in jobs]
    else:
        with multiprocessing.Pool(workers) as pool:
            raw = pool.map(_play_job, jobs)

    scores, wins = [], 0.0
    for i, (seating, episode_scores, winners) in enumerate(raw):
        seat = 0 if seats == 1 else i % seats
        scores.append(episode_scores[seat])
        if seat in winners:
            wins += 1.0 / len(winners)

    mean = statistics.fmean(scores)
    stderr = statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
    rate, low, high = wilson_interval(wins, len(scores))
    return {
        "mean_score": mean,
        "stderr": stderr,
        "score_low": mean - 1.96 * stderr,
        "score_high": mean + 1.96 * stderr,
        "win_rate": rate,
        "win_low": low,
        "win_high": high,
        "games": len(scores),
    }


def sweep(kind, game, vary, values, *, base=None, prefix=None, players=None,
          arena_games=400, arena_opponent="random", verbose=True, **fixed):
    """Train one model per value of `vary` and evaluate them identically.

    `base`/`**fixed` are parameters held constant across the sweep and passed
    to the trainer. Anything the trainer accepts works; a typo raises rather
    than being silently ignored.

    `arena_games`/`arena_opponent` control the *measurement* afterwards, not the
    training. They are named distinctly on purpose: the DQN trainer has its own
    `eval_games`, and a sweep argument that quietly captured it would configure
    the wrong thing.

    Returns a list of row dicts, one per value, ready for a DataFrame.
    """
    train = _trainer(kind)
    params = {**(base or {}), **fixed}
    if players is not None:
        params["players"] = players
    prefix = prefix or f"{kind}-{vary}"

    rows = []
    for value in values:
        name = f"{prefix}-{str(value).replace('/', '_')}"
        if verbose:
            print(f"training {name} …", end=" ", flush=True)

        started = time.time()
        # sweep=True writes under models/<game>/sweeps/, which is gitignored:
        # these are disposable experiment output, not curated models.
        result = train(game, **{**params, vary: value, "name": name,
                                "save": True, "sweep": True, "quiet": True})
        trained = time.time() - started

        scored = evaluate_model(
            game, name, games=arena_games, opponent=arena_opponent, players=players
        )
        row = {
            vary: value,
            "model": name,
            "train_seconds": round(trained, 1),
            **scored,
        }
        # Keep the trainer's own progress metric so training and evaluation can
        # be compared -- they disagree more often than you would expect.
        if kind == "ga":
            row["holdout_score"] = result["holdout_score"]
        else:
            row["final_eval"] = result["eval_score"]
            row["best_eval"] = result["best_eval"]
        rows.append(row)

        if verbose:
            print(
                f"{trained:5.1f}s train · score {scored['mean_score']:8.2f} "
                f"± {1.96 * scored['stderr']:.2f} · win {scored['win_rate']:.1%}"
            )
    return rows


def round_robin(game, names, *, games=600, players=2, seed_offset=7_000_000,
                workers=None, verbose=False):
    """Every model against every other. Returns nested dict of win rates.

    `matrix[a][b]` is a's win rate against b. Seats rotate within each pairing,
    and every pairing replays the same seeds.
    """
    kwargs = {"num_players": players}
    matrix = {a: {} for a in names}
    intervals = {a: {} for a in names}

    for i, a in enumerate(names):
        for b in names:
            if a == b:
                matrix[a][b] = float("nan")
                intervals[a][b] = (float("nan"), float("nan"))
                continue
            if b in matrix and a in matrix[b] and not math.isnan(matrix[b].get(a, float("nan"))):
                continue  # already played from the other side

            jobs = []
            for k in range(games):
                seating = (a, b) if k % 2 == 0 else (b, a)
                jobs.append((game, kwargs, seating, seed_offset + k))

            workers = workers or multiprocessing.cpu_count()
            with multiprocessing.Pool(workers) as pool:
                raw = pool.map(_play_job, jobs)

            wins_a = 0.0
            for seating, _scores, winners in raw:
                seat_a = seating.index(a)
                if seat_a in winners:
                    wins_a += 1.0 / len(winners)

            rate, low, high = wilson_interval(wins_a, games)
            matrix[a][b] = rate
            matrix[b][a] = 1.0 - rate
            intervals[a][b] = (low, high)
            intervals[b][a] = (1.0 - high, 1.0 - low)
            if verbose:
                print(f"{a} vs {b}: {rate:.1%} [{low:.1%}, {high:.1%}]")

    return {"win_rate": matrix, "interval": intervals, "games": games}


def separable(low_a, high_a, low_b, high_b):
    """True when two intervals do not overlap, i.e. the ranking is established."""
    return high_a < low_b or high_b < low_a
