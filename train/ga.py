"""Genetic algorithm over the per-action feature weights of any Game.

Because `action_features` lives on the game, this trainer is game-agnostic: it
evolves a `WeightedAgent` for 2048, Snake, or anything else added later.

Three things here matter more than the GA itself:

1. **Common random numbers.** Every individual in a generation is scored on the
   same seeds, so a better agent is not confused with a luckier one.
2. **A games-per-eval ramp.** Early generations are cheap and noisy, which is
   fine while differences are large; later generations pay for precision, which
   is when it is needed. A flat budget wastes effort at one end or the other.
3. **Fresh seeds each generation**, plus a fixed holdout used only for
   reporting, so the progress curve is comparable across generations.
"""

from __future__ import annotations

import argparse
import multiprocessing
import random
import statistics
import time
from types import SimpleNamespace

from core.agent import WeightedAgent
from core.registry import load_agent, make_game, save_model
from core.runner import play_episode

# Defined once and shared by the CLI and `train_ga`, so the two cannot drift.
DEFAULTS = {
    "name": "ga",
    "players": None,
    "opponent": "random",
    "population": 40,
    "generations": 30,
    "games": 12,
    "games_end": 60,
    "holdout_games": 40,
    "elites": 4,
    "tournament": 3,
    "sigma": 0.35,
    "sigma_decay": 0.97,
    "mutation_rate": 0.3,
    "workers": multiprocessing.cpu_count(),
    "seed": 0,
    "save": True,
    "sweep": False,
    "quiet": False,
}


def train_ga(game, **overrides):
    """Programmatic entry point. Returns a result dict; see `run`.

    Keyword names match the CLI flags with dashes turned into underscores.
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameter(s): {sorted(unknown)}")
    return run(SimpleNamespace(game=game, **{**DEFAULTS, **overrides}))


def _fitness(job):
    """Mean score of one weight vector over a fixed seed list."""
    weights, game_name, game_kwargs, seeds, opponent = job
    game = make_game(game_name, **game_kwargs)
    candidate = WeightedAgent(weights, name="candidate")
    total = 0.0
    for i, seed in enumerate(seeds):
        if game.num_players == 1:
            seat, agents = 0, [candidate]
        else:
            # Rotate which seat the candidate occupies so seat bias cancels.
            seat = i % game.num_players
            agents = [load_agent(game_name, opponent) for _ in range(game.num_players)]
            agents[seat] = candidate
        total += play_episode(game, agents, seed).scores[seat]
    return total / len(seeds)


def evaluate_population(pool, population, game_name, game_kwargs, seeds, opponent):
    jobs = [(ind, game_name, game_kwargs, seeds, opponent) for ind in population]
    return [_fitness(j) for j in jobs] if pool is None else pool.map(_fitness, jobs)


def tournament(population, fitnesses, rng, size):
    best = rng.randrange(len(population))
    for _ in range(size - 1):
        challenger = rng.randrange(len(population))
        if fitnesses[challenger] > fitnesses[best]:
            best = challenger
    return population[best]


def crossover(parent_a, parent_b, rng, alpha=0.4):
    """Blend crossover. Unlike a single-point cut it can produce values outside
    the parents' range, so the population does not collapse onto its initial spread."""
    child = []
    for a, b in zip(parent_a, parent_b):
        low, high = min(a, b), max(a, b)
        spread = (high - low) * alpha
        child.append(rng.uniform(low - spread, high + spread))
    return tuple(child)


def mutate(individual, rng, sigma, rate):
    return tuple(
        gene + rng.gauss(0.0, sigma) if rng.random() < rate else gene
        for gene in individual
    )


def run(args):
    game_kwargs = {}
    if args.players is not None:
        game_kwargs["num_players"] = args.players
    game = make_game(args.game, **game_kwargs)
    n_features = len(game.feature_names)

    rng = random.Random(args.seed)
    population = [
        tuple(rng.gauss(0.0, 1.0) for _ in range(n_features))
        for _ in range(args.population)
    ]
    holdout = [rng.randrange(2**31) for _ in range(args.holdout_games)]

    pool = None if args.workers == 1 else multiprocessing.Pool(args.workers)
    history = []
    started = time.time()

    try:
        for generation in range(1, args.generations + 1):
            # Ramp precision as the population converges and gaps narrow.
            fraction = (generation - 1) / max(1, args.generations - 1)
            budget = round(args.games + fraction * (args.games_end - args.games))
            seeds = [rng.randrange(2**31) for _ in range(budget)]

            fitnesses = evaluate_population(
                pool, population, args.game, game_kwargs, seeds, args.opponent
            )
            ranked = sorted(zip(fitnesses, population), key=lambda p: -p[0])
            best_fitness, best = ranked[0]
            holdout_fitness = _fitness(
                (best, args.game, game_kwargs, holdout, args.opponent)
            )

            history.append({
                "generation": generation,
                "games_per_eval": budget,
                "best": best_fitness,
                "mean": statistics.fmean(fitnesses),
                "holdout": holdout_fitness,
                "weights": list(best),
            })
            if not args.quiet:
                print(
                    f"gen {generation:3d}  n={budget:4d}  "
                    f"best {best_fitness:9.1f}  mean {statistics.fmean(fitnesses):9.1f}  "
                    f"holdout {holdout_fitness:9.1f}  [{time.time() - started:5.0f}s]",
                    flush=True,
                )

            survivors = [ind for _, ind in ranked[: args.elites]]
            sigma = args.sigma * (args.sigma_decay ** (generation - 1))
            while len(survivors) < args.population:
                a = tournament(population, fitnesses, rng, args.tournament)
                b = tournament(population, fitnesses, rng, args.tournament)
                survivors.append(mutate(crossover(a, b, rng), rng, sigma, args.mutation_rate))
            population = survivors
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    champion = max(history, key=lambda h: h["holdout"])
    if not args.quiet:
        print(f"\nbest holdout {champion['holdout']:.1f} at generation {champion['generation']}")
        for name, weight in zip(game.feature_names, champion["weights"]):
            print(f"  {name:<14} {weight:+.3f}")

    payload = {
        "kind": "weighted",
        "weights": champion["weights"],
        "feature_names": list(game.feature_names),
        "holdout_score": champion["holdout"],
        "generation": champion["generation"],
        "config": vars(args),
        "history": history,
    }
    path = None
    if args.save:
        path = save_model(args.game, args.name, payload, sweep=args.sweep)
        if not args.quiet:
            print(f"wrote {path}")

    return {
        "name": args.name,
        "game": args.game,
        "weights": champion["weights"],
        "holdout_score": champion["holdout"],
        "history": history,
        "path": str(path) if path else None,
        "seconds": time.time() - started,
        "config": vars(args),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game")
    parser.add_argument("--name", default=DEFAULTS["name"], help="model name to save under")
    parser.add_argument("--players", type=int, default=DEFAULTS["players"])
    parser.add_argument("--opponent", default=DEFAULTS["opponent"],
                        help="agent filling the other seats in multiplayer games")
    parser.add_argument("--population", type=int, default=DEFAULTS["population"])
    parser.add_argument("--generations", type=int, default=DEFAULTS["generations"])
    parser.add_argument("--games", type=int, default=DEFAULTS["games"],
                        help="games per individual in generation 1")
    parser.add_argument("--games-end", type=int, default=DEFAULTS["games_end"],
                        help="games per individual in the final generation")
    parser.add_argument("--holdout-games", type=int, default=DEFAULTS["holdout_games"])
    parser.add_argument("--elites", type=int, default=DEFAULTS["elites"])
    parser.add_argument("--tournament", type=int, default=DEFAULTS["tournament"])
    parser.add_argument("--sigma", type=float, default=DEFAULTS["sigma"])
    parser.add_argument("--sigma-decay", type=float, default=DEFAULTS["sigma_decay"])
    parser.add_argument("--mutation-rate", type=float, default=DEFAULTS["mutation_rate"])
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.save = True
    args.sweep = False
    run(args)


if __name__ == "__main__":
    main()
