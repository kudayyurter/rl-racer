"""Torch networks shared by the gradient-based methods (REINFORCE, A2C, PPO).

Why this exists: all three methods learn the same kind of policy, a small tanh network that
outputs the *mean* of a Gaussian over [steer, throttle]. Only the learning rule differs, so the
networks and the plumbing the viewer needs (``torch_activations``, ``torch_layer_weights``)
live here, while each algorithm's rollout, returns and loss stay in its own module.

The policy is stochastic while training (sample around the mean, which explores) and
deterministic when driving for real (use the mean, which is what ``evaluate`` and the viewer do).
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_INIT = -0.5  # std = exp(-0.5) ~ 0.61: wide enough to explore, narrow enough to steer


def _mlp(sizes: list[int]) -> nn.Sequential:
    """Linear layers with tanh between them (none after the last)."""
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Policy: observation -> Normal over actions.

    The mean comes from a tanh MLP; the standard deviation is a learned parameter that does not
    depend on the observation. Actions are sampled *unclipped*; the environment clips them to
    [-1, 1]. Log-probabilities use the unclipped sample, so the gradient stays exact.
    """

    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.mean(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    """State-value function V(obs) -> (B,). Unused by REINFORCE; A2C and PPO add it."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def torch_activations(module: nn.Module, obs: np.ndarray) -> list[np.ndarray]:
    """For the viewer: [input, each hidden layer's post-tanh output, final output] as numpy.

    ``module`` is anything holding an ``nn.Sequential`` called ``net`` (actor or critic).
    """
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    acts = [x.numpy().astype(float)]
    with torch.no_grad():
        for layer in module.net:
            x = layer(x)
            if isinstance(layer, nn.Tanh) or layer is module.net[-1]:
                acts.append(x.numpy().astype(float))
    return acts


def torch_layer_weights(module: nn.Module) -> list[np.ndarray]:
    """Weight matrices shaped (in, out). nn.Linear stores (out, in), hence the transpose."""
    return [layer.weight.detach().numpy().T.astype(float)
            for layer in module.net if isinstance(layer, nn.Linear)]
