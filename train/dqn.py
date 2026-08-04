"""DQN training loop.

Works against any Game. In a multiplayer game the learner takes one seat and a
fixed opponent fills the rest; seats rotate so it does not learn a position.

Uses Double DQN: the online network picks the next action, the target network
scores it. Vanilla DQN takes a max over its own noisy estimates, which biases
Q-values upward — a few extra characters here remove that.

**n-step returns** (`--n-step`, 1 for textbook DQN). Instead of one reward plus
a bootstrap, a transition carries the discounted sum of the next n rewards and
bootstraps from n decisions later. The point is how fast reward information
travels backwards: at n=1 a snake that dies teaches only the state it died in,
and the move that actually trapped it — ten ticks earlier — has to wait for ten
separate rounds of bootstrapping before it hears anything. At n=3 that is three
rounds. In sparse-reward games this is usually the single largest win available.

The honest caveat is that it is **biased off-policy**. The intermediate actions
in an n-step return came from whatever policy was current when the episode was
played, not the one being trained, and nothing here corrects for that. The bias
grows with n, which is why 3 is the usual sweet spot and why large n gets worse
rather than continuing to get better. `notebooks/optimize.ipynb` measures where
that turn happens for these games instead of assuming it.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.dqn import ConvNet, DQNAgent, MLP, ReplayBuffer
from core.game import NO_ACTION, SIMULTANEOUS
from core.registry import load_agent, make_game, save_model
from core.runner import play_episode

# Gradient-norm clip. Early Q-targets are wild, and an unclipped step off one
# can undo a lot of training.
CLIP = 10.0

# Defined once and shared by the CLI and `train_dqn`, so the two cannot drift.
DEFAULTS = {
    "name": "dqn",
    "players": None,
    "opponent": "random",
    "episodes": 4000,
    # Observation encoding, and with it the architecture: a flat observation
    # gets an MLP, a grid one gets a conv tower. `game.encodings` lists what a
    # game offers; only Snake currently offers more than one.
    "encoding": "flat",
    "hidden": 128,
    # Conv tower shape. Ignored unless the encoding produces a grid.
    "channels": 64,
    "blocks": 3,
    "reduce": 32,
    "n_step": 1,
    # Episodes played concurrently. Purely a throughput setting -- the same
    # number of episodes and gradient steps happen either way -- but it decides
    # whether the network sees batches or single positions while collecting.
    "envs": 1,
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


def _emit(window, buffer, next_obs, done, legal_next, gamma):
    """Store the n-step transition starting at the oldest decision in `window`.

    `window` holds `[obs, action, reward]` per decision, oldest first, where the
    reward is everything that accrued between that decision and the next one.
    The stored return is the discounted sum across the whole window, and the
    bootstrap it will be paired with is `next_obs`, `len(window)` decisions
    later. Storing the length is what lets `train_step` discount by the right
    power of gamma for a window cut short at the end of an episode.
    """
    obs, action, _ = window[0]
    total, discount = 0.0, 1.0
    for _, _, reward in window:
        total += discount * reward
        discount *= gamma
    buffer.add(obs, action, total, next_obs, done, legal_next, len(window))


def _act_batch(agent, game, views, players, rngs, observations):
    """`agent.act_batch` if it has one, otherwise a loop over `act`.

    `core/runner.py` only ever requires `act`, so an agent written against that
    contract must keep working here. Subclassing `core.agent.Agent` gets the
    same loop for free; this covers anything that does not.
    """
    batched = getattr(agent, "act_batch", None)
    if batched is not None:
        return batched(game, views, players, rngs, observations=observations)
    return [agent.act(game, v, p, r) for v, p, r in zip(views, players, rngs)]


def collect_batch(game, learner, opponent_name, seeds, buffer, seats,
                  n_step=1, gamma=1.0):
    """Play one episode per seed *concurrently*. Returns the learner's scores.

    Transitions run **from one of the learner's decisions to its next one**, not
    from tick to tick. In a simultaneous game those are the same thing and this
    is the textbook loop. In a turn-based game they are not: the opponent moves
    in between, so the state the learner bootstraps from has to be the position
    it actually faces next, and the reward has to be everything that accrued
    while it was not on move. Storing per-tick instead would have the learner
    bootstrapping from positions where it is not even to move, and in Connect
    Four -- where the only reward is the final one, and it lands on the
    *opponent's* ply when they win -- it would never see a loss at all.

    With `n_step` > 1 a transition spans n decisions rather than one, so the
    window below trails n decisions behind play and each one is emitted only
    once its successor is known. Counting in *decisions* rather than ticks is
    the same choice as above, and it is what makes n-step mean the same thing
    in a turn-based game as in a simultaneous one.

    **Why the episodes run together.** Stepping N Python games costs exactly N
    times stepping one; nothing about the game gets faster. What changes is that
    every agent is asked for all of its pending decisions at once, so a network
    does one forward pass over N positions instead of N passes over one. At
    batch 1 a GPU spends more time launching the kernel than running it, so this
    is the difference between using the device and merely owning it.

    Episodes finish at different lengths, so the batch shrinks as it goes and
    the last few ticks are nearly serial. That is unavoidable without resetting
    finished environments mid-batch, which would break the one-seed-one-episode
    correspondence the arena and the tests rely on.

    **One opponent object serves the whole batch**, every seat and every episode,
    rather than one being loaded per episode as the serial path did. Two reasons
    it is safe: no agent here keeps state between decisions -- `reset` is a no-op
    for all of them -- and every decision still draws from its own episode's and
    seat's rng, which is what the equivalence tests check. It is also the point:
    one object means one group below, so all of the opponent's decisions across
    the batch become a single forward pass instead of one per seat, and a model
    is read from disk once instead of N times.
    """
    if len(seeds) != len(seats):
        raise ValueError(f"got {len(seeds)} seeds but {len(seats)} seats")

    encoding = getattr(learner, "encoding", None)
    opponent = None
    if seats and game.num_players > 1:
        opponent = load_agent(game.name, opponent_name)
        opponent.reset()

    learner.reset()
    episodes = []
    for seed, seat in zip(seeds, seats):
        rng = random.Random(seed)
        episodes.append(SimpleNamespace(
            rng=rng,
            agents=[learner if i == seat else opponent
                    for i in range(game.num_players)],
            agent_rngs=[random.Random(seed * 7919 + i)
                        for i in range(game.num_players)],
            seat=seat,
            state=game.reset(rng),
            window=deque(),   # up to n_step [obs, action, reward], oldest first
            accrued=0.0,      # reward since the most recent decision
            obs=None,         # observation at this tick's decision, if any
            actions=None,
        ))

    live = [e for e in episodes if not game.is_terminal(e.state)]
    while live:
        # Group pending decisions by the agent that owns them, so each agent's
        # network is called once for the whole batch. Grouping by agent rather
        # than by seat is what makes this work under seat rotation: the learner
        # sits in different seats in different episodes but is one object.
        groups = {}
        for e in live:
            turn = game.current_player(e.state)
            movers = range(game.num_players) if turn == SIMULTANEOUS else (turn,)
            e.actions = [NO_ACTION] * game.num_players
            for i in movers:
                if game.legal_actions(e.state, i):
                    agent = e.agents[i]
                    groups.setdefault(id(agent), (agent, []))[1].append((e, i))

        for agent, rows in groups.values():
            views = [game.view(e.state, i) for e, i in rows]
            players = [i for _, i in rows]
            rngs = [e.agent_rngs[i] for e, i in rows]

            # The learner's observations are needed for the replay buffer
            # anyway, so build them here and hand them to the agent rather than
            # letting it build the same vectors a second time.
            observations = None
            if agent is learner:
                observations = [game.observe(v, p, encoding)
                                for v, p in zip(views, players)]
                for (e, _), obs in zip(rows, observations):
                    e.obs = obs

            picks = _act_batch(agent, game, views, players, rngs, observations)
            for (e, i), pick in zip(rows, picks):
                e.actions[i] = pick

        for e in live:
            options = game.legal_actions(e.state, e.seat)
            if options:
                if e.window:
                    # Close out the previous decision's reward interval, and
                    # now that we know where it led, emit anything aged out.
                    e.window[-1][2] = e.accrued
                    if len(e.window) == n_step:
                        _emit(e.window, buffer, e.obs, False, options, gamma)
                        e.window.popleft()
                e.window.append([e.obs, e.actions[e.seat], 0.0])
                e.accrued = 0.0

            next_state = game.step(e.state, tuple(e.actions), e.rng)
            e.accrued += game.reward(e.state, next_state, e.seat)
            e.state = next_state

        live = [e for e in live
                if not game.is_terminal(e.state) and game.alive(e.state, e.seat)]

    scores = []
    for e in episodes:
        if e.window:
            e.window[-1][2] = e.accrued
            final = game.observe(game.view(e.state, e.seat), e.seat, encoding)
            # The episode ended, so there is nothing to bootstrap from and every
            # decision still in the window gets a true return to the end of the
            # episode. `done` zeroes the bootstrap term, which is also why the
            # window being shorter than n_step here does not matter.
            while e.window:
                _emit(e.window, buffer, final, True, [], gamma)
                e.window.popleft()
        scores.append(game.scores(e.state)[e.seat])
    return scores


def collect_episode(game, learner, opponent_name, seed, buffer, seat,
                    n_step=1, gamma=1.0):
    """One episode. A batch of one, so there is only ever one implementation."""
    return collect_batch(game, learner, opponent_name, [seed], buffer, [seat],
                         n_step=n_step, gamma=gamma)[0]


def train_step(net, target, opt, buffer, batch_size, gamma, n_actions):
    obs, actions, rewards, next_obs, done, legal, nsteps = buffer.sample(batch_size)
    # The buffer already stores each column in the dtype torch wants, so these
    # are plain copies onto the device.
    obs, actions, rewards, next_obs, done, nsteps = (
        torch.as_tensor(x, device=net.device)
        for x in (obs, actions, rewards, next_obs, done, nsteps)
    )
    legal = torch.as_tensor(legal[:, :n_actions], device=net.device)

    with torch.no_grad():
        # Double DQN: online net chooses, target net values.
        online_next = net(next_obs).masked_fill(~legal, -torch.inf)
        best = online_next.argmax(dim=1, keepdim=True)
        bootstrap = target(next_obs).gather(1, best).squeeze(1)
        # A state with no legal follow-up contributes nothing beyond its reward.
        bootstrap = torch.where(legal.any(dim=1), bootstrap, 0.0)
        # `rewards` is already the discounted n-step sum, so the bootstrap it
        # meets is n decisions further away and discounted to match. At n=1
        # this is gamma, i.e. exactly the one-step target.
        targets = rewards + gamma ** nsteps * (1.0 - done) * bootstrap

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


def build_for(game, args):
    """Pick an architecture from what the chosen encoding produces.

    Returns `(make_net, obs_size, obs_dtype)`. `make_net` is called twice --
    once for the online net, once for the target -- so it has to be a factory
    rather than a net to copy, or the two would share initialisation only by
    accident of `load_state_dict` running afterwards.
    """
    obs_size, grid = game.obs_spec(args.encoding)

    if grid is None:
        sizes = [obs_size, args.hidden, args.hidden, game.num_actions]
        return (lambda: MLP(sizes)), obs_size, np.float32

    scalars = obs_size - grid[0] * grid[1] * grid[2]
    make = lambda: ConvNet(  # noqa: E731 - a factory, see above
        grid, scalars, game.num_actions,
        channels=args.channels, blocks=args.blocks,
        reduce=args.reduce, hidden=args.hidden,
    )
    # Grid observations are large enough that float32 storage dominates memory
    # long before the network does. See ReplayBuffer's docstring.
    return make, obs_size, np.float16


def run(args):
    game_kwargs = {"num_players": args.players} if args.players else {}
    game = make_game(args.game, **game_kwargs)
    if args.encoding not in game.encodings:
        raise ValueError(
            f"{args.game} has no encoding {args.encoding!r}; "
            f"have {list(game.encodings)}"
        )
    if args.n_step < 1:
        raise ValueError(f"n_step must be at least 1, got {args.n_step}")
    if args.envs < 1:
        raise ValueError(f"envs must be at least 1, got {args.envs}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch reports no CUDA device")

    torch.manual_seed(args.seed)
    make_net, obs_size, obs_dtype = build_for(game, args)
    net = make_net().to(args.device)
    target = make_net().to(args.device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    learner = DQNAgent(net, name=args.name, epsilon=args.epsilon_start,
                       encoding=args.encoding)
    buffer = ReplayBuffer(args.buffer, obs_size, seed=args.seed, dtype=obs_dtype)
    rng = random.Random(args.seed)
    eval_seeds = [900_000 + i for i in range(args.eval_games)]

    history = []
    started = time.time()
    updates = 0
    params = sum(p.numel() for p in net.parameters())

    if not args.quiet:
        print(
            f"{args.game}/{args.name}: {args.encoding} obs ({obs_size} floats) · "
            f"{net.arch['kind']} {params:,} params · n_step {args.n_step} · "
            f"{args.envs} env{'s' if args.envs > 1 else ''} · "
            f"{args.device} · buffer {buffer.nbytes / 1e6:.0f} MB",
            flush=True,
        )

    done = 0
    while done < args.episodes:
        # The last batch is short when `envs` does not divide `episodes`, so the
        # total number of episodes played is exactly what was asked for however
        # the two are set. That keeps `--envs` a pure throughput knob: it must
        # not quietly change how much experience a run is trained on.
        batch = min(args.envs, args.episodes - done)
        episode = done + batch

        # Linear decay to a small floor: explore hard early, exploit later.
        progress = min(1.0, episode / (args.episodes * args.explore_fraction))
        learner.epsilon = args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)

        seeds = [rng.randrange(2**31) for _ in range(batch)]
        seats = [(done + k + 1) % game.num_players for k in range(batch)]
        scores = collect_batch(
            game, learner, args.opponent, seeds, buffer, seats,
            n_step=args.n_step, gamma=args.gamma,
        )
        score = statistics.fmean(scores)

        losses = []
        if buffer.size >= args.warmup:
            # Updates scale with the experience just collected, so the replay
            # ratio -- gradient steps per new transition -- is what `--envs`
            # holds fixed. Raising `envs` without this would quietly train a
            # tenth as much and look like a speedup.
            for _ in range(args.updates_per_episode * batch):
                losses.append(train_step(
                    net, target, opt, buffer, args.batch, args.gamma,
                    game.num_actions,
                ))
                updates += 1
                if updates % args.target_sync == 0:
                    target.load_state_dict(net.state_dict())

        done = episode
        if done % args.eval_every < batch or done == args.episodes:
            mean_score = evaluate(game, learner, args.opponent, eval_seeds)
            history.append({
                "episode": done,
                "epsilon": learner.epsilon,
                "train_score": score,
                "eval_score": mean_score,
                "loss": statistics.fmean(losses) if losses else None,
                "buffer": buffer.size,
            })
            if not args.quiet:
                print(
                    f"ep {done:6d}  eps {learner.epsilon:.3f}  "
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
        "params": params,
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
