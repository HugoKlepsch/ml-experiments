"""A DQN built on torch.

Two architectures, both ending in a linear head with one Q-value per action:

* `MLP` -- ReLU stack over a flat observation. The original, and still the
  right choice for a game whose observation is a handful of numbers.
* `ConvNet` -- a conv tower over a `(C, H, W)` board plus a scalar
  side-channel, for games that offer a grid encoding. Used by Snake's
  `planes`; see `games/snake.py` for what the planes are.

Trained with Huber loss, Adam, a target network and a uniform replay buffer --
the training loop itself lives in `train/dqn.py`.

Model files are JSON rather than `torch.save` output, holding the module's
`state_dict` as nested lists alongside the config and training history the
notebooks plot. That keeps every model in `models/` -- GA and DQN alike -- one
readable, diffable format that `core/registry.py` can load without knowing which
kind it is until it reads the file. Each net records an `arch` dict that is
exactly its constructor arguments, so `build_net` can rebuild it without the
loader knowing which architecture it is about to get.
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


def _he_init(module):
    """He initialisation on every conv and linear layer in `module`.

    Variance 2/fan_in keeps ReLU activations from collapsing toward zero as
    depth grows -- which matters much more for the conv tower than it did for a
    two-layer MLP.
    """
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Fully connected net with ReLU hidden layers and a linear output.

    `sizes` is [obs, hidden..., actions]; any depth works.
    """

    def __init__(self, sizes):
        super().__init__()
        self.sizes = list(sizes)
        self.arch = {"kind": "mlp", "sizes": self.sizes}

        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers[:-1])  # the head is linear, so drop its ReLU
        _he_init(self)

    def forward(self, x):
        return self.net(x)

    @property
    def device(self):
        return next(self.parameters()).device


class ConvNet(nn.Module):
    """Conv tower over a board, with the scalar features concatenated at the head.

    Input is the flat vector `observe` returns: `C*H*W` grid entries followed by
    `scalars` more. Splitting it here rather than in the replay buffer is what
    keeps the buffer a plain 2-D array.

    Three deliberate choices:

    * **No pooling.** Pooling throws away where things are, and in Snake where
      the food is *is* the problem. Every conv keeps the board at H x W with
      `padding=1`, so the tower's receptive field grows by one cell per layer
      and position survives to the head.
    * **A 1x1 convolution before flattening.** Flattening 64 channels of a
      12x12 board into a 128-wide layer is 1.2M parameters, most of the net.
      Squeezing to `reduce` channels first cuts that severalfold for no
      measurable loss -- the same trick AlphaZero's policy head uses.
    * **No normalisation layers.** BatchNorm interacts badly with a target
      network, which sees different batch statistics than the online net and so
      disagrees with it for reasons that have nothing to do with learning.

    `blocks` is the depth that matters: with 3x3 convs the tower sees a
    (2*blocks+1)-square window, so 3 blocks reach 7 cells and anything beyond
    that has to come through the scalar channel.
    """

    def __init__(self, grid, scalars, n_actions, channels=64, blocks=3,
                 reduce=32, hidden=128):
        super().__init__()
        self.arch = {
            "kind": "conv", "grid": list(grid), "scalars": scalars,
            "n_actions": n_actions, "channels": channels, "blocks": blocks,
            "reduce": reduce, "hidden": hidden,
        }
        self.grid = tuple(grid)
        self.scalars = scalars
        planes, height, width = self.grid
        self.grid_size = planes * height * width

        tower, in_channels = [], planes
        for _ in range(blocks):
            tower += [nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU()]
            in_channels = channels
        tower += [nn.Conv2d(in_channels, reduce, 1), nn.ReLU()]
        self.tower = nn.Sequential(*tower)

        self.head = nn.Sequential(
            nn.Linear(reduce * height * width + scalars, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        _he_init(self)

    def forward(self, x):
        if x.shape[-1] != self.grid_size + self.scalars:
            raise ValueError(
                f"expected {self.grid_size + self.scalars} inputs, got {x.shape[-1]}; "
                "the agent's encoding and the game's are out of step"
            )
        board = x[:, :self.grid_size].view(-1, *self.grid)
        scalars = x[:, self.grid_size:]
        features = self.tower(board).flatten(1)
        return self.head(torch.cat([features, scalars], dim=1))

    @property
    def device(self):
        return next(self.parameters()).device


def build_net(arch):
    """Rebuild a network from the `arch` dict it saved."""
    arch = dict(arch)
    kind = arch.pop("kind", "mlp")
    if kind == "mlp":
        return MLP(arch["sizes"])
    if kind == "conv":
        return ConvNet(**arch)
    raise ValueError(f"unknown network kind {kind!r}")


class DQNAgent(Agent):
    """Greedy over predicted Q-values, restricted to legal actions.

    `epsilon` is only non-zero during training; a saved agent plays greedily.

    The agent remembers the observation `encoding` it was trained on and asks
    the game for that one. This is what lets a flat agent and a planes agent
    play each other in the arena off a single game object -- the alternative,
    making encoding a property of the game, would mean the arena had to know
    what every seat was trained on before it could build the board.
    """

    kind = "dqn"

    def __init__(self, net, name="dqn", epsilon=0.0, encoding=None):
        self.net = net
        self.name = name
        self.epsilon = epsilon
        self.encoding = encoding

    @torch.no_grad()
    def q_values(self, game, state, player):
        obs = torch.tensor(game.observe(state, player, self.encoding),
                           dtype=torch.float32, device=self.net.device)
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

    @torch.no_grad()
    def act_batch(self, game, states, players, rngs, observations=None):
        """One forward pass for every position that needs one.

        Two things this must get exactly right, or a vectorised run stops
        matching a serial one:

        * **rng draws.** `act` calls `rng.random()` only when epsilon is
          non-zero -- `self.epsilon and ...` short-circuits otherwise -- and
          `rng.choice` only when that draw lands inside epsilon. The loop below
          does the same, per row, in row order.
        * **Rows that explored are not sent to the network.** They already have
          an action, and including them would change nothing except the batch
          contents. Skipping them means early training, where epsilon is near
          1, costs almost no forward passes at all.
        """
        chosen = [NO_ACTION] * len(states)
        rows = []
        for k, (state, player, rng) in enumerate(zip(states, players, rngs)):
            options = game.legal_actions(state, player)
            if not options:
                continue
            if self.epsilon and rng.random() < self.epsilon:
                chosen[k] = rng.choice(options)
            else:
                rows.append((k, options))

        if rows:
            batch = [
                observations[k] if observations is not None
                else game.observe(states[k], players[k], self.encoding)
                for k, _ in rows
            ]
            q = self.net(torch.tensor(batch, dtype=torch.float32,
                                      device=self.net.device)).tolist()
            for (k, options), values in zip(rows, q):
                chosen[k] = max(options, key=lambda a: values[a])
        return chosen

    def to_payload(self):
        return {
            "kind": self.kind,
            "encoding": self.encoding,
            "arch": self.net.arch,
            "state_dict": {k: v.tolist() for k, v in self.net.state_dict().items()},
        }

    @classmethod
    def from_payload(cls, payload, name="dqn"):
        arch = payload.get("arch")
        if arch is None:
            # Written before there was more than one architecture: an MLP,
            # described by its layer sizes and nothing else.
            arch = {"kind": "mlp", "sizes": payload["sizes"]}
        net = build_net(arch)
        net.load_state_dict({
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in payload["state_dict"].items()
        })
        net.eval()
        return cls(net, name=name, encoding=payload.get("encoding"))


class ReplayBuffer:
    """Fixed-capacity uniform replay.

    Breaking the correlation between consecutive frames is the point: training
    on a trajectory in order makes the updates wildly non-independent and the
    network chases its own tail.

    Kept in numpy rather than as a preallocated tensor because it is written one
    row at a time from the collector and only ever read a batch at a time; the
    copy to torch happens once per update, in `train_step`.

    `nsteps` records how many of the learner's decisions each transition spans,
    so `train_step` knows to discount its bootstrap by `gamma ** nsteps`. It is
    1 for ordinary DQN and can be shorter than the configured n at the end of an
    episode, where there are not n decisions left to accumulate.

    `dtype` is the storage type for observations only. A 12x12 board of 8 planes
    is 1171 floats, and at the default capacity two float32 copies of that is
    940 MB -- so grid encodings store float16 instead and cast on the way out.
    Observations are inputs, not parameters: half precision costs about three
    decimal digits of something already normalised to roughly [0, 1].
    """

    def __init__(self, capacity, obs_size, seed=0, dtype=np.float32):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_size), dtype=dtype)
        self.next_obs = np.zeros((capacity, obs_size), dtype=dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.nsteps = np.ones(capacity, dtype=np.float32)
        self.legal = np.zeros((capacity, 8), dtype=bool)
        self.size = 0
        self.cursor = 0

    @property
    def nbytes(self):
        """Resident size, which is worth knowing before you raise `buffer`."""
        return sum(
            a.nbytes for a in
            (self.obs, self.next_obs, self.actions, self.rewards, self.done,
             self.nsteps, self.legal)
        )

    def add(self, obs, action, reward, next_obs, done, legal_next, nsteps=1):
        i = self.cursor
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.nsteps[i] = nsteps
        self.legal[i] = False
        for a in legal_next:
            self.legal[i, a] = True
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, size=batch)
        # astype is a no-op when storage is already float32, and the one place
        # a float16 buffer pays for itself back.
        return (
            self.obs[idx].astype(np.float32, copy=False),
            self.actions[idx], self.rewards[idx],
            self.next_obs[idx].astype(np.float32, copy=False),
            self.done[idx], self.legal[idx], self.nsteps[idx],
        )
