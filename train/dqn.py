"""DQN training loop.

Works against any Game. In a multiplayer game the learner takes one seat and a
fixed opponent fills the rest; seats rotate so it does not learn a position.

Uses Double DQN: the online network picks the next action, the target network
scores it. Vanilla DQN takes a max over its own noisy estimates, which biases
Q-values upward — a few extra characters here remove that.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from agents.dqn import DQNAgent, MLP, ReplayBuffer
from core.registry import load_agent, make_game, save_model
from core.runner import play_episode, poll

# Gradient-norm clip. Early Q-targets are wild, and an unclipped step off one
# can undo a lot of training.
CLIP = 10.0

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
    # These nets are far too small to pay back a GPU: at batch 64 over a
    # 128-wide MLP, kernel launch overhead dominates the arithmetic. Set
    # --device cuda only if you have grown the network enough to matter.
    "device": "cpu",
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


def train_step(net, target, opt, buffer, batch_size, gamma, n_actions):
    obs, actions, rewards, next_obs, done, legal = buffer.sample(batch_size)
    # The buffer already stores each column in the dtype torch wants, so these
    # are plain copies onto the device.
    obs, actions, rewards, next_obs, done = (
        torch.as_tensor(x, device=net.device)
        for x in (obs, actions, rewards, next_obs, done)
    )
    legal = torch.as_tensor(legal[:, :n_actions], device=net.device)

    with torch.no_grad():
        # Double DQN: online net chooses, target net values.
        online_next = net(next_obs).masked_fill(~legal, -torch.inf)
        best = online_next.argmax(dim=1, keepdim=True)
        bootstrap = target(next_obs).gather(1, best).squeeze(1)
        # A state with no legal follow-up contributes nothing beyond its reward.
        bootstrap = torch.where(legal.any(dim=1), bootstrap, 0.0)
        targets = rewards + gamma * (1.0 - done) * bootstrap

    predicted = net(obs).gather(1, actions[:, None]).squeeze(1)
    # smooth_l1_loss at the default beta=1 is the Huber loss: linear near zero,
    # clipped past 1. The clipping matters because early Q-targets are wildly
    # wrong, and squared error on those produces updates large enough to
    # destabilise training.
    loss = F.smooth_l1_loss(predicted, targets)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), CLIP)
    opt.step()
    return float((predicted.detach() - targets).abs().mean())


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

    torch.manual_seed(args.seed)
    sizes = [game.obs_size, args.hidden, args.hidden, game.num_actions]
    net = MLP(sizes).to(args.device)
    target = MLP(sizes).to(args.device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

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
                    net, target, opt, buffer, args.batch, args.gamma,
                    game.num_actions,
                ))
                updates += 1
                if updates % args.target_sync == 0:
                    target.load_state_dict(net.state_dict())

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
