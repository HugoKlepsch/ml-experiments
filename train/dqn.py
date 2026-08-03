"""DQN training loop.

Works against any Game. In a multiplayer game the learner takes one seat and a
fixed opponent fills the rest; seats rotate so it does not learn a position.

Uses Double DQN: the online network picks the next action, the target network
scores it. Vanilla DQN takes a max over its own noisy estimates, which biases
Q-values upward — a few extra characters here remove that.
"""

from __future__ import annotations

import argparse
import multiprocessing
import random
import statistics
import time
from types import SimpleNamespace

import numpy as np

from agents.dqn import DQNAgent, MLP, ReplayBuffer, huber_grad
from core.registry import load_agent, make_game, save_model
from core.runner import play_episode, poll

# Defined once and shared by the CLI and `train_dqn`, so the two cannot drift.
DEFAULTS = {
    "name": "dqn",
    "players": None,
    "opponent": "random",
    "episodes": 4000,
    "hidden": 128,
    "lr": 7e-4,
    "gamma": 0.95,
    "batch": 64,
    "buffer": 100_000,
    "warmup": 1_000,
    "updates_per_episode": 8,
    "target_sync": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "explore_fraction": 0.5,
    "eval_every": 200,
    "eval_games": 30,
    "seed": 0,
    "save": True,
    "sweep": False,
    "quiet": False,
}


def train_dqn(game, **overrides):
    """Programmatic entry point. Returns a result dict; see `run`.

    Keyword names match the CLI flags with dashes turned into underscores.
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameter(s): {sorted(unknown)}")
    return run(SimpleNamespace(game=game, **{**DEFAULTS, **overrides}))


def collect_episode(game, learner, opponent_name, seed, buffer, seat):
    """Play one episode, storing the learner's transitions. Returns its score.

    Transitions run **from one of the learner's decisions to its next one**, not
    from tick to tick. In a simultaneous game those are the same thing and this
    is the textbook loop. In a turn-based game they are not: the opponent moves
    in between, so the state the learner bootstraps from has to be the position
    it actually faces next, and the reward has to be everything that accrued
    while it was not on move. Storing per-tick instead would have the learner
    bootstrapping from positions where it is not even to move, and in Connect
    Four -- where the only reward is the final one, and it lands on the
    *opponent's* ply when they win -- it would never see a loss at all.
    """
    rng = random.Random(seed)
    agents = [learner if i == seat else load_agent(game.name, opponent_name)
              for i in range(game.num_players)]
    agent_rngs = [random.Random(seed * 7919 + i) for i in range(game.num_players)]
    for agent in agents:
        agent.reset()

    state = game.reset(rng)
    pending = None    # (obs, action) from the learner's last decision
    accrued = 0.0     # reward since that decision

    while not game.is_terminal(state):
        options = game.legal_actions(state, seat)
        actions = poll(game, agents, state, agent_rngs)
        if options:
            # A decision point: close out the previous one, then open this one.
            obs = game.observe(game.view(state, seat), seat)
            if pending is not None:
                buffer.add(*pending, accrued, obs, False, options)
            pending, accrued = (obs, actions[seat]), 0.0

        next_state = game.step(state, tuple(actions), rng)
        accrued += game.reward(state, next_state, seat)
        state = next_state

        if not game.alive(state, seat):
            break

    if pending is not None:
        # The episode ended without the learner moving again, so there is
        # nothing to bootstrap from: `done` zeroes the bootstrap term.
        buffer.add(*pending, accrued, game.observe(game.view(state, seat), seat), True, [])
    return game.scores(state)[seat]


def train_step(net, target, buffer, batch_size, gamma, lr, n_actions):
    obs, actions, rewards, next_obs, done, legal = buffer.sample(batch_size)
    legal = legal[:, :n_actions]

    # Double DQN: online net chooses, target net values.
    online_next = net.predict(next_obs)
    online_next = np.where(legal, online_next, -np.inf)
    best = online_next.argmax(axis=1)
    target_next = target.predict(next_obs)
    bootstrap = target_next[np.arange(len(best)), best]
    # A state with no legal follow-up contributes nothing beyond its reward.
    bootstrap = np.where(legal.any(axis=1), bootstrap, 0.0)

    targets = rewards + gamma * (1.0 - done) * bootstrap

    q, cache = net.forward(obs)
    rows = np.arange(len(actions))
    predicted = q[rows, actions]
    grad = huber_grad(predicted, targets) / len(actions)

    d_out = np.zeros_like(q)
    d_out[rows, actions] = grad
    net.adam_step(net.backward(cache, d_out), lr=lr)
    return float(np.abs(predicted - targets).mean())


def evaluate(game, agent, opponent_name, seeds):
    """Greedy score on fixed seeds, so the curve is comparable across time."""
    saved, agent.epsilon = agent.epsilon, 0.0
    scores = []
    try:
        for i, seed in enumerate(seeds):
            seat = i % game.num_players
            agents = [load_agent(game.name, opponent_name)
                      for _ in range(game.num_players)]
            agents[seat] = agent
            scores.append(play_episode(game, agents, seed).scores[seat])
    finally:
        agent.epsilon = saved
    return statistics.fmean(scores)


def run(args):
    game_kwargs = {"num_players": args.players} if args.players else {}
    game = make_game(args.game, **game_kwargs)

    net = MLP([game.obs_size, args.hidden, args.hidden, game.num_actions], seed=args.seed)
    target = MLP([game.obs_size, args.hidden, args.hidden, game.num_actions], seed=args.seed)
    target.copy_from(net)

    learner = DQNAgent(net, name=args.name, epsilon=args.epsilon_start)
    buffer = ReplayBuffer(args.buffer, game.obs_size, seed=args.seed)
    rng = random.Random(args.seed)
    eval_seeds = [900_000 + i for i in range(args.eval_games)]

    history = []
    started = time.time()
    updates = 0

    for episode in range(1, args.episodes + 1):
        # Linear decay to a small floor: explore hard early, exploit later.
        progress = min(1.0, episode / (args.episodes * args.explore_fraction))
        learner.epsilon = args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)

        seat = episode % game.num_players
        score = collect_episode(
            game, learner, args.opponent, rng.randrange(2**31), buffer, seat
        )

        losses = []
        if buffer.size >= args.warmup:
            for _ in range(args.updates_per_episode):
                losses.append(train_step(
                    net, target, buffer, args.batch, args.gamma, args.lr,
                    game.num_actions,
                ))
                updates += 1
                if updates % args.target_sync == 0:
                    target.copy_from(net)

        if episode % args.eval_every == 0 or episode == args.episodes:
            mean_score = evaluate(game, learner, args.opponent, eval_seeds)
            history.append({
                "episode": episode,
                "epsilon": learner.epsilon,
                "train_score": score,
                "eval_score": mean_score,
                "loss": statistics.fmean(losses) if losses else None,
                "buffer": buffer.size,
            })
            if not args.quiet:
                print(
                    f"ep {episode:6d}  eps {learner.epsilon:.3f}  "
                    f"loss {statistics.fmean(losses) if losses else float('nan'):8.3f}  "
                    f"eval {mean_score:9.1f}  [{time.time() - started:5.0f}s]",
                    flush=True,
                )

    learner.epsilon = 0.0
    payload = {
        **learner.to_payload(),
        "config": vars(args),
        "history": history,
    }
    path = None
    if args.save:
        path = save_model(args.game, args.name, payload, sweep=args.sweep)
        if not args.quiet:
            print(f"\nwrote {path}")

    return {
        "name": args.name,
        "game": args.game,
        "eval_score": history[-1]["eval_score"] if history else float("nan"),
        "best_eval": max((h["eval_score"] for h in history), default=float("nan")),
        "history": history,
        "path": str(path) if path else None,
        "seconds": time.time() - started,
        "config": vars(args),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game")
    for flag, default in DEFAULTS.items():
        if flag in ("save", "quiet"):
            continue
        kind = type(default) if default is not None else int
        parser.add_argument(
            f"--{flag.replace('_', '-')}",
            type=kind if kind in (int, float) else str,
            default=default,
        )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.save = True
    args.sweep = False
    run(args)


if __name__ == "__main__":
    main()
