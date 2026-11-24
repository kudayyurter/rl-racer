"""Lessons 05-07: what does a Gaussian actor's *mean* look like, before the env clips it?

For each checkpoint given (REINFORCE, A2C, PPO or SB3 PPO; anything with a Gaussian actor, see
``actor_of.py``), prints:
  - the unclipped mean [steer, throttle] on the start line, and the learned std,
  - over the states of the deterministic evaluation episode (what ``evaluate`` drives): the
    range of each mean and the share outside [-1, 1], plus how the episode went,
  - over training-like states (16 cars sampling actions for 512 decisions each, auto-reset,
    seed 0): the same range and share outside [-1, 1].
The env clips actions to [-1, 1], so a mean of 3 drives exactly like a mean of 1. This script
shows how far outside the range a trained mean has wandered.

Run: uv run python docs/lessons/scripts/policy_means.py runs/a2c-<timestamp>-s0 100 200
     (iteration numbers of checkpoints; ``latest`` for checkpoints/latest.pt)
"""

import json
import sys

import numpy as np
import torch

from actor_of import gaussian_actor
from racer.config import EnvConfig, from_dict
from racer.policy import load_policy
from racer.sim import BatchSim
from racer.track import load_track


def describe(means: np.ndarray) -> str:
    parts = []
    for k, name in enumerate(("steer", "throttle")):
        m = means[:, k]
        parts.append(f"{name} [{m.min():+5.2f}, {m.max():+5.2f}] "
                     f"(<-1: {(m < -1).mean():4.0%}, >1: {(m > 1).mean():4.0%})")
    return "  ".join(parts)


def main() -> None:
    run, which = sys.argv[1], sys.argv[2:]
    config = json.loads(open(f"{run}/config.json").read())
    env = from_dict(EnvConfig, config["env"])
    track = load_track(config["train"]["track"])
    torch.manual_seed(0)
    for w in which:
        name = "latest.pt" if w == "latest" else f"ckpt_{int(w):05d}.pt"
        actor = gaussian_actor(load_policy(f"{run}/checkpoints/{name}"))

        def mean(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return actor.mean(torch.as_tensor(obs, dtype=torch.float32)).numpy()

        # Deterministic evaluation episode, exactly as evaluate() drives it.
        sim = BatchSim(track, env, n_cars=1, seed=0, auto_reset=False)
        obs = sim.reset()
        start = mean(obs)[0]
        eval_means = []
        while sim.alive[0]:
            m = mean(obs)
            eval_means.append(m[0])
            obs = sim.step(np.clip(m, -1.0, 1.0)).obs
        ended = f"{float(sim.episode_time[0]):.1f} s, {float(sim.traveled[0]):.1f} m, " \
                f"{int(sim.laps[0])} laps"

        # Training-like states: sample from the Gaussian, as collection does.
        sim = BatchSim(track, env, n_cars=16, seed=0, auto_reset=True)
        obs = sim.reset()
        train_means = []
        for _ in range(512):
            with torch.no_grad():
                dist = actor.dist(torch.as_tensor(obs, dtype=torch.float32))
            train_means.append(dist.mean.numpy())
            obs = sim.step(dist.sample().numpy()).obs
        train_means = np.concatenate(train_means)

        std = actor.log_std.detach().exp().numpy()
        print(f"{name}: start-line mean steer {start[0]:+.3f} throttle {start[1]:+.3f}; "
              f"std steer {std[0]:.3f} throttle {std[1]:.3f}")
        print(f"  eval episode ({ended}, {len(eval_means)} decisions): "
              f"{describe(np.array(eval_means))}")
        print(f"  training-like states ({len(train_means):,}): {describe(train_means)}")


if __name__ == "__main__":
    main()
