"""Lesson 05: how noisy is one REINFORCE gradient estimate?

Loads a REINFORCE checkpoint and collects several independent batches with it, exactly as
training does (``collect_batch``, reward-to-go, ``reinforce_loss``). For each batch it computes
the gradient of the loss with respect to all actor parameters, without stepping. If the
estimates were precise, every batch would point the same way. It prints:
  - the mean cosine similarity between gradients of different batches (1 = identical direction,
    0 = unrelated, as two random directions in this many dimensions would be),
  - |mean gradient| / mean |gradient|: near 1 when the batches agree, near 1/sqrt(K) when they
    are pure noise around a zero mean.
Run it with two ``episodes_per_batch`` values to see how batch size trades updates for noise.

Run: uv run python docs/lessons/scripts/gradient_noise.py runs/reinforce-<timestamp>-s0 25 32 8
     (run, checkpoint iteration, episodes per batch, number of batches)
"""

import json
import sys

import numpy as np
import torch

from racer.algos.reinforce import collect_batch, discounted_returns, reinforce_loss
from racer.config import EnvConfig, from_dict
from racer.policy import load_policy
from racer.sim import BatchSim
from racer.track import load_track


def main() -> None:
    run, it, n_episodes, n_batches = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    config = json.loads(open(f"{run}/config.json").read())
    env = from_dict(EnvConfig, config["env"])
    gamma = config["train"]["gamma"]
    track = load_track(config["train"]["track"])
    actor = load_policy(f"{run}/checkpoints/ckpt_{it:05d}.pt").actor
    torch.set_num_threads(2)
    n_params = sum(p.numel() for p in actor.parameters())

    grads, steps = [], []
    for b in range(n_batches):
        torch.manual_seed(1000 + b)
        sim = BatchSim(track, env, n_cars=16, seed=1000 + b, auto_reset=True)
        episodes, n = collect_batch(sim, actor, n_episodes)
        returns = np.concatenate([discounted_returns(r, gamma) for _, _, r in episodes])
        obs = torch.as_tensor(np.concatenate([o for o, _, _ in episodes]), dtype=torch.float32)
        act = torch.as_tensor(np.concatenate([a for _, a, _ in episodes]), dtype=torch.float32)
        actor.zero_grad()
        logp = actor.dist(obs).log_prob(act).sum(-1)
        reinforce_loss(logp, torch.as_tensor(returns, dtype=torch.float32)).backward()
        grads.append(torch.cat([p.grad.flatten() for p in actor.parameters()]).numpy().copy())
        steps.append(n)

    g = np.array(grads)
    unit = g / np.linalg.norm(g, axis=1, keepdims=True)
    cos = unit @ unit.T
    pairs = cos[np.triu_indices(n_batches, k=1)]
    ratio = np.linalg.norm(g.mean(0)) / np.linalg.norm(g, axis=1).mean()
    print(f"{run} checkpoint {it}: {n_batches} batches of {n_episodes} episodes "
          f"(mean {np.mean(steps):,.0f} decisions simulated per batch), {n_params:,} parameters")
    print(f"  cosine similarity between batch gradients: mean {pairs.mean():+.3f}, "
          f"min {pairs.min():+.3f}, max {pairs.max():+.3f}")
    print(f"  |mean gradient| / mean |gradient| = {ratio:.3f}  "
          f"(pure noise would give about 1/sqrt({n_batches}) = {1 / np.sqrt(n_batches):.3f})")


if __name__ == "__main__":
    main()
