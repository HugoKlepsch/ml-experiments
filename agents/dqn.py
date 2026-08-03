"""A DQN written directly in numpy, backward pass included.

Torch would be fewer lines, but the whole point of this project is to see how
the model works. The network here is small enough that hand-written backprop is
readable and fast, and swapping in torch later only touches this file.

Architecture: MLP with ReLU hidden layers and a linear head producing one
Q-value per action. Trained with Huber loss, Adam, a target network and a
uniform replay buffer.
"""

from __future__ import annotations

import numpy as np

from core.agent import Agent
from core.game import NO_ACTION


class MLP:
    """Fully connected net with ReLU hidden layers and a linear output."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.weights = [
            # He initialisation: variance 2/fan_in keeps ReLU activations from
            # collapsing toward zero as depth grows.
            rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b))
            for a, b in zip(sizes, sizes[1:])
        ]
        self.biases = [np.zeros(b) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.weights + self.biases]
        self._v = [np.zeros_like(p) for p in self.weights + self.biases]
        self._t = 0

    def forward(self, x):
        """Returns (output, cache). `x` has shape (batch, in)."""
        activations = [x]
        pre = []
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            pre.append(z)
            activations.append(z if i == last else np.maximum(z, 0.0))
        return activations[-1], (activations, pre)

    def predict(self, x):
        return self.forward(x)[0]

    def backward(self, cache, d_out):
        """Gradients for a loss whose derivative w.r.t. the output is `d_out`."""
        activations, pre = cache
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        delta = d_out
        for i in reversed(range(len(self.weights))):
            grads_w[i] = activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (pre[i - 1] > 0)
        return grads_w + grads_b

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip=10.0):
        params = self.weights + self.biases
        self._t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            norm = np.linalg.norm(grad)
            if norm > clip:
                grad = grad * (clip / norm)
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * grad
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * grad * grad
            m_hat = self._m[i] / (1 - beta1 ** self._t)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other):
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def to_lists(self):
        return {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    @classmethod
    def from_lists(cls, payload):
        net = cls(payload["sizes"])
        net.weights = [np.array(w, dtype=float) for w in payload["weights"]]
        net.biases = [np.array(b, dtype=float) for b in payload["biases"]]
        return net


class DQNAgent(Agent):
    """Greedy over predicted Q-values, restricted to legal actions.

    `epsilon` is only non-zero during training; a saved agent plays greedily.
    """

    kind = "dqn"

    def __init__(self, net, name="dqn", epsilon=0.0):
        self.net = net
        self.name = name
        self.epsilon = epsilon

    def q_values(self, game, state, player):
        obs = np.asarray(game.observe(state, player), dtype=float)[None, :]
        return self.net.predict(obs)[0]

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
        return {"kind": self.kind, "net": self.net.to_lists()}

    @classmethod
    def from_payload(cls, payload, name="dqn"):
        return cls(MLP.from_lists(payload["net"]), name=name)


class ReplayBuffer:
    """Fixed-capacity uniform replay.

    Breaking the correlation between consecutive frames is the point: training
    on a trajectory in order makes the updates wildly non-independent and the
    network chases its own tail.
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


def huber_grad(predicted, target, delta=1.0):
    """d/dpred of Huber loss: linear near zero, clipped past `delta`.

    Clipping matters here because early Q-targets are wildly wrong, and squared
    error on those produces updates large enough to destabilise training.
    """
    diff = predicted - target
    return np.clip(diff, -delta, delta)
