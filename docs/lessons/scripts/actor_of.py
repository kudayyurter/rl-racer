"""Lesson 07 helper: the Gaussian actor inside a loaded policy, ours or Stable-Baselines3's.

The lesson scripts (``policy_means.py``, ``noisy_vs_mean_laps.py``) need the policy's
*unclipped* mean and its sampling distribution, not just the clipped deterministic action that
``Policy.act`` returns. Our REINFORCE / A2C / PPO policies expose that as ``policy.actor`` (a
``GaussianActor``). An SB3 policy keeps the same pieces under other names: the actor MLP is
``mlp_extractor.policy_net``, the mean layer is ``action_net`` and the state-independent log std
is ``policy.log_std``. ``SB3Actor`` wraps them behind the ``GaussianActor`` interface
(``mean``, ``dist``, ``log_std``) so the scripts treat both the same way.

Not a test fixture and not used by training: only the lesson scripts import it.
"""

import torch
from torch.distributions import Normal


class SB3Actor:
    """An SB3 ``ActorCriticPolicy``'s actor with ``GaussianActor``'s interface."""

    def __init__(self, sb3_policy):
        self.policy = sb3_policy
        self.log_std = sb3_policy.log_std

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        # MlpPolicy's feature extractor is a plain Flatten, so the observation goes straight in.
        return self.policy.action_net(self.policy.mlp_extractor.forward_actor(obs))

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.mean(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))


def gaussian_actor(policy):
    """``policy.actor`` for our methods; an ``SB3Actor`` for an SB3 policy (``racer.algos.sb3_ppo``)."""
    if hasattr(policy, "actor"):
        return policy.actor
    return SB3Actor(policy.model.policy)
