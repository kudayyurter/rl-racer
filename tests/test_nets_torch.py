"""Tests for the torch networks and the Policy plumbing (racer/nets_torch.py)."""

import numpy as np
import torch
from torch.distributions import Normal

from racer.algos.reinforce import TorchPolicy, discounted_returns
from racer.nets_torch import Critic, GaussianActor, torch_layer_weights


def test_discounted_returns():
    got = discounted_returns(np.array([1.0, 1.0, 1.0]), 0.5)
    assert np.allclose(got, [1.75, 1.5, 1.0])


def test_logprob_matches_distribution():
    torch.manual_seed(0)
    actor = GaussianActor(5, 2, (8,))
    obs, a = torch.randn(4, 5), torch.randn(4, 2)
    mean, std = actor.mean(obs), actor.log_std.exp()
    by_dim = Normal(mean[:, 0], std[0]).log_prob(a[:, 0]) + Normal(mean[:, 1], std[1]).log_prob(a[:, 1])
    assert torch.allclose(actor.dist(obs).log_prob(a).sum(-1), by_dim, atol=1e-6)


def test_critic_shape():
    assert Critic(5, (8, 8))(torch.zeros(3, 5)).shape == (3,)


def test_policy_plumbing():
    actor = GaussianActor(5, 2, (8, 6))
    policy = TorchPolicy(actor)
    obs = np.random.default_rng(0).normal(size=(1, 5))
    acts = policy.activations(obs)
    assert [a.shape for a in acts] == [(1, 5), (1, 8), (1, 6), (1, 2)]
    assert [w.shape for w in torch_layer_weights(actor)] == [(5, 8), (8, 6), (6, 2)]
    assert np.all(np.abs(acts[1]) <= 1)                       # post-tanh hidden
    assert np.allclose(acts[-1], actor.mean(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy(), atol=1e-6)
    big = policy.act(np.full((3, 5), 100.0))
    assert big.shape == (3, 2) and np.all(np.abs(big) <= 1)   # deterministic and clipped
