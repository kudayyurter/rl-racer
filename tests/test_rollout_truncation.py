"""Truncation wiring in A2C's and PPO's collect_rollout (lesson 06's headline fix).

``test_truncation_bootstrap`` in test_a2c.py checks the arithmetic of add_truncation_bootstrap.
This checks what feeds it: collect_rollout must ask the critic about ``final_obs`` (the state the
car was really in, not the respawn) on truncated steps only, and record 0 on terminated steps and
ordinary ones, so a crash keeps its "no future" and a time limit does not.
"""

import numpy as np
import pytest
import torch
from torch import nn

from racer.algos import a2c, ppo
from racer.nets_torch import GaussianActor
from racer.sim import StepResult


class FirstFeatureCritic(nn.Module):
    """V(obs) = obs[:, 0], so a recorded value tells which observation the critic was shown."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, 0]


class ScriptedSim:
    """Stands in for BatchSim: replays fixed StepResults for three cars.

    Step 1: car 0 is truncated (time limit), car 1 is terminated (crash), car 2 drives on.
    Both finished cars respawn, so their ``obs`` (100) differs from ``final_obs`` (5 and 7).
    Step 2: nobody finishes.
    """

    n_cars = 3

    def __init__(self):
        no, yes = False, True
        self.results = iter([
            StepResult(obs=np.array([[100.0, 0], [100.0, 0], [9.0, 0]]), reward=np.ones(3),
                       terminated=np.array([no, yes, no]), truncated=np.array([yes, no, no]),
                       lap_completed=np.zeros(3, bool), lap_time=np.full(3, np.nan),
                       final_obs=np.array([[5.0, 0], [7.0, 0], [9.0, 0]])),
            StepResult(obs=np.array([[101.0, 0], [101.0, 0], [10.0, 0]]), reward=np.ones(3),
                       terminated=np.zeros(3, bool), truncated=np.zeros(3, bool),
                       lap_completed=np.zeros(3, bool), lap_time=np.full(3, np.nan),
                       final_obs=np.array([[101.0, 0], [101.0, 0], [10.0, 0]])),
        ])

    def step(self, actions):
        return next(self.results)


@pytest.mark.parametrize("algo", [a2c, ppo], ids=["a2c", "ppo"])
def test_bootstrap_only_on_truncated_steps_and_from_final_obs(algo):
    torch.manual_seed(0)
    batch, _ = algo.collect_rollout(ScriptedSim(), GaussianActor(2, 2, (4,)), FirstFeatureCritic(),
                                    np.zeros((3, 2)), n_steps=2)
    # V(final_obs) = 5 for the truncated car (not V(respawn) = 100); 0 for the crash and the
    # car still driving; 0 everywhere on the step where nobody finished.
    np.testing.assert_array_equal(batch["final_values"], [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rewards = algo.add_truncation_bootstrap(batch["rewards"], batch["truncated"],
                                            batch["final_values"], 0.9)
    np.testing.assert_allclose(rewards, [[1.0 + 0.9 * 5.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
