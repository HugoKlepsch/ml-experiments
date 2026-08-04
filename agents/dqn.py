"""A DQN built on torch.

Architecture: MLP with ReLU hidden layers and a linear head producing one
Q-value per action. Trained with Huber loss, Adam, a target network and a
uniform replay buffer -- the training loop itself lives in `train/dqn.py`.

Model files are JSON rather than `torch.save` output, holding the module's
`state_dict` as nested lists alongside the config and training history the
notebooks plot. That keeps every model in `models/` -- GA and DQN alike -- one
readable, diffable format that `core/registry.py` can load without knowing which
kind it is until it reads the file.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core.agent import Agent
from core.game import NO_ACTION

# The arena and the sweeps fan out over a process pool, and each worker would
# otherwise start its own set of intra-op threads: on a 16-core box that is 256
# threads fighting over 128x128 matmuls. One thread per process is both faster
# here and avoids the well-known torch-threadpool-across-fork hang.
torch.set_num_threads(1)


class MLP(nn.Module):
    """Fully connected net with ReLU hidden layers and a linear output.

    `sizes` is [obs, hidden..., actions]; any depth works.
    """

    def __init__(self, sizes):
        super().__init__()
        self.sizes = list(sizes)

        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers[:-1])  # the head is linear, so drop its ReLU

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # He initialisation: variance 2/fan_in keeps ReLU activations
                # from collapsing toward zero as depth grows.
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

    @property
    def device(self):
        return next(self.parameters()).device


class DQNAgent(Agent):
    """Greedy over predicted Q-values, restricted to legal actions.

    `epsilon` is only non-zero during training; a saved agent plays greedily.
    """

    kind = "dqn"

    def __init__(self, net, name="dqn", epsilon=0.0):
        self.net = net
        self.name = name
        self.epsilon = epsilon

    @torch.no_grad()
    def q_values(self, game, state, player):
        obs = torch.tensor(game.observe(state, player), dtype=torch.float32,
                           device=self.net.device)
        return self.net(obs.unsqueeze(0))[0].tolist()

    def act(self, game, state, player, rng):
        options = game.legal_actions(state, player)
        if not options:
            return NO_ACTION
        if self.epsilon and rng.random() < self.epsilon:
            return rng.choice(options)
        q = self.q_values(game, state, player)
        # Masking rather than penalising: an illegal action must never be
        # chosen, no matter what the network predicts for it.
        return max(options, key=lambda a: q[a])

    def to_payload(self):
        return {
            "kind": self.kind,
            "sizes": self.net.sizes,
            "state_dict": {k: v.tolist() for k, v in self.net.state_dict().items()},
        }

    @classmethod
    def from_payload(cls, payload, name="dqn"):
        net = MLP(payload["sizes"])
        net.load_state_dict({
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in payload["state_dict"].items()
        })
        net.eval()
        return cls(net, name=name)


class ReplayBuffer:
    """Fixed-capacity uniform replay.

    Breaking the correlation between consecutive frames is the point: training
    on a trajectory in order makes the updates wildly non-independent and the
    network chases its own tail.

    Kept in numpy rather than as a preallocated tensor because it is written one
    row at a time from the collector and only ever read a batch at a time; the
    copy to torch happens once per update, in `train_step`.
    """

    def __init__(self, capacity, obs_size, seed=0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.legal = np.zeros((capacity, 8), dtype=bool)
        self.size = 0
        self.cursor = 0

    def add(self, obs, action, reward, next_obs, done, legal_next):
        i = self.cursor
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.legal[i] = False
        for a in legal_next:
            self.legal[i, a] = True
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, size=batch)
        return (
            self.obs[idx], self.actions[idx], self.rewards[idx],
            self.next_obs[idx], self.done[idx], self.legal[idx],
        )
