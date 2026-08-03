"""DQN training loop.

Works against any Game. In a multiplayer game the learner takes one seat and a
fixed opponent fills the rest; seats rotate so it does not learn a position.

Uses Double DQN: the online network picks the next action, the target network
scores it. Vanilla DQN takes a max over its own noisy estimates, which biases
Q-values upward — a few extra characters here remove that.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import numpy as np

from agents.dqn import DQNAgent, MLP, ReplayBuffer, huber_grad
from core.game import NO_ACTION
from core.registry import load_agent, make_game, save_model
from core.runner import play_episode


def collect_episode(game, learner, opponent_name, seed, buffer, seat):
    """Play one episode, storing the learner's transitions. Returns its score."""
    rng = random.Random(seed)
    agents = [None] * game.num_players
    for i in range(game.num_players):
        agents[i] = learner if i == seat else load_agent(game.name, opponent_name)
    agent_rng = random.Random(seed * 7919)
    for agent in agents:
        if agent is not None:
            agent.reset()

    state = game.reset(rng)
    while not game.is_terminal(state):
        options = game.legal_actions(state, seat)
        if not options:
            break
        obs = game.observe(state, seat)
        action = learner.act(game, state, seat, agent_rng)

        actions = []
        for i, agent in enumerate(agents):
            if i == seat:
                actions.append(action)
            elif game.legal_actions(state, i):
                actions.append(agent.act(game, state, i, agent_rng))
            else:
                actions.append(NO_ACTION)

        next_state = game.step(state, tuple(actions), rng)
        reward = game.reward(state, next_state, seat)
        done = game.is_terminal(next_state) or not game.alive(next_state, seat)
        legal_next = [] if done else game.legal_actions(next_state, seat)
        buffer.add(obs, action, reward, game.observe(next_state, seat), done, legal_next)

        state = next_state
        if done:
            break
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
            print(
                f"ep {episode:6d}  eps {learner.epsilon:.3f}  "
                f"loss {statistics.fmean(losses) if losses else float('nan'):8.3f}  "
                f"eval {mean_score:9.1f}  [{time.time() - started:5.0f}s]",
                flush=True,
            )

    learner.epsilon = 0.0
    path = save_model(args.game, args.name, {
        **learner.to_payload(),
        "config": vars(args),
        "history": history,
    })
    print(f"\nwrote {path}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game")
    parser.add_argument("--name", default="dqn")
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--target-sync", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--explore-fraction", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
