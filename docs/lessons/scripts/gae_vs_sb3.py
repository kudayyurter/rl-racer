"""Lesson 07: does our ``compute_gae`` give the same advantages as Stable-Baselines3's?

Feeds the same random rollout (rewards, values, episode ends, last values) to our
``racer.algos.ppo.compute_gae`` and to SB3's ``RolloutBuffer.compute_returns_and_advantage``,
for several lambdas, and prints the largest absolute difference in the advantages and in the
critic targets (returns). SB3 marks episode ends differently: it stores ``episode_start[t]``
(the step *after* an end) and takes the final step's ``dones`` separately, so we convert.
Truncation bootstrap is not part of either GAE routine (both fold gamma * V(final_obs) into the
reward beforehand), so plain random rewards are enough here.

Run: uv run python docs/lessons/scripts/gae_vs_sb3.py
"""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from racer.algos.ppo import compute_gae

T, N, GAMMA = 128, 16, 0.99


def sb3_gae(rewards, values, dones, last_values, gamma, lam):
    buf = RolloutBuffer(T, spaces.Box(-1, 1, (1,)), spaces.Box(-1, 1, (1,)), device="cpu",
                        gae_lambda=lam, gamma=gamma, n_envs=N)
    episode_start = np.ones(N)  # SB3 convention: the first step of a rollout counts as a start
    for t in range(T):
        buf.add(np.zeros((N, 1)), np.zeros((N, 1)), rewards[t], episode_start,
                torch.as_tensor(values[t]), torch.zeros(N))
        episode_start = dones[t].astype(float)
    buf.compute_returns_and_advantage(torch.as_tensor(last_values), dones[-1].astype(float))
    return buf.advantages, buf.returns


def main() -> None:
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = rng.random((T, N)) < 0.05
    last_values = rng.normal(size=N).astype(np.float32)
    for lam in (0.0, 0.9, 0.95, 1.0):
        ours = compute_gae(rewards, values, dones, last_values, GAMMA, lam)
        adv, ret = sb3_gae(rewards, values, dones, last_values, GAMMA, lam)
        print(f"lam {lam:4.2f}: max |A_ours - A_sb3| = {np.abs(ours - adv).max():.2e}, "
              f"max |R_ours - R_sb3| = {np.abs(ours + values - ret).max():.2e}  "
              f"({T} x {N} steps, {int(dones.sum())} episode ends, advantages up to "
              f"{np.abs(ours).max():.1f})")


if __name__ == "__main__":
    main()
