"""Head-to-head evaluation with honest error bars.

Two habits are baked in here because leaving them out is how you fool yourself:

* **Common random numbers.** Every agent faces the same list of seeds. For
  one-player games that makes the comparison *paired*, which shrinks the error
  bar enormously because the shared luck cancels.
* **Seat rotation.** In multiplayer, agents are cycled through every seat so
  that a positional advantage cannot be mistaken for a better model.

Win rates are reported with Wilson intervals rather than bare percentages. A
55% win rate over 40 games and over 4000 games are very different claims.
"""

from __future__ import annotations

import math
import multiprocessing
import statistics
from dataclasses import dataclass, field

from core.registry import load_agent, make_game
from core.runner import play_episode


def wilson_interval(successes, total, z=1.96):
    """Confidence interval for a proportion. Behaves sensibly at 0% and 100%,
    which the normal approximation does not."""
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class AgentResult:
    name: str
    games: int = 0
    wins: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    scores: list = field(default_factory=list)

    @property
    def win_rate(self):
        return wilson_interval(self.wins, self.games)

    def line(self, multiplayer=True):
        if multiplayer:
            rate, low, high = self.win_rate
            return (
                f"  {self.name:<18} win {rate:6.1%}  "
                f"[{low:5.1%}, {high:5.1%}]   mean score {self.mean_score:9.1f}"
            )
        return (
            f"  {self.name:<18} mean {self.mean_score:9.1f}  "
            f"median {self.median_score:9.1f}  best {max(self.scores):9.1f}"
        )


def _play_job(job):
    game_name, game_kwargs, seating, seed = job
    game = make_game(game_name, **game_kwargs)
    agents = [load_agent(game_name, n) for n in seating]
    episode = play_episode(game, agents, seed)
    return seating, episode.scores, episode.winners


def compare(game_name, agent_names, games=200, seed_offset=1_000_000,
            workers=None, game_kwargs=None, quiet=False):
    """Compare agents on one game. Returns {agent_name: AgentResult}."""
    game_kwargs = dict(game_kwargs or {})
    probe = make_game(game_name, **game_kwargs)
    players = probe.num_players
    multiplayer = players > 1

    if multiplayer and len(agent_names) != players:
        raise ValueError(
            f"{game_name} seats {players} players, got {len(agent_names)} agents"
        )

    seeds = [seed_offset + i for i in range(games)]
    jobs = []
    if multiplayer:
        for i, s in enumerate(seeds):
            # Rotate seats each game so seat bias averages out.
            shift = i % len(agent_names)
            seating = tuple(agent_names[(shift + k) % len(agent_names)]
                            for k in range(len(agent_names)))
            jobs.append((game_name, game_kwargs, seating, s))
    else:
        for name in agent_names:
            jobs.extend((game_name, game_kwargs, (name,), s) for s in seeds)

    workers = workers or multiprocessing.cpu_count()
    if workers == 1:
        raw = [_play_job(j) for j in jobs]
    else:
        with multiprocessing.Pool(workers) as pool:
            raw = pool.map(_play_job, jobs)

    results = {name: AgentResult(name=name) for name in agent_names}
    for seating, scores, winners in raw:
        for seat, name in enumerate(seating):
            record = results[name]
            record.games += 1
            record.scores.append(scores[seat])
            # A tie splits the win so rates across agents still sum to 1.
            if seat in winners:
                record.wins += 1.0 / len(winners)

    for record in results.values():
        record.mean_score = statistics.fmean(record.scores)
        record.median_score = statistics.median(record.scores)

    if not quiet:
        _report(game_name, results, agent_names, games, multiplayer)
    return results


def _report(game_name, results, agent_names, games, multiplayer):
    label = "games each" if not multiplayer else "games, seats rotated"
    print(f"\n{game_name}: {games} {label}")
    for name in agent_names:
        print(results[name].line(multiplayer))

    if not multiplayer and len(agent_names) == 2:
        a, b = (results[n] for n in agent_names)
        _paired_report(a, b)


def _paired_report(a, b):
    """Paired difference on shared seeds: the tight comparison CRN buys us."""
    diffs = [x - y for x, y in zip(a.scores, b.scores)]
    mean = statistics.fmean(diffs)
    if len(diffs) < 2:
        return
    stderr = statistics.stdev(diffs) / math.sqrt(len(diffs))
    low, high = mean - 1.96 * stderr, mean + 1.96 * stderr
    verdict = "significant" if low > 0 or high < 0 else "NOT significant"
    print(
        f"\n  paired difference ({a.name} - {b.name}): "
        f"{mean:+.1f}  95% CI [{low:+.1f}, {high:+.1f}]  -> {verdict}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare agents head to head.")
    parser.add_argument("game")
    parser.add_argument("agents", nargs="+")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=None,
                        help="seat count for multiplayer games")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.players is not None:
        kwargs["num_players"] = args.players
    elif args.game == "snake":
        kwargs["num_players"] = len(args.agents)

    compare(args.game, args.agents, games=args.games,
            workers=args.workers, game_kwargs=kwargs)


if __name__ == "__main__":
    main()
