"""Lessons 05-06: does the trained *noisy* policy drive differently from its *mean*?

Training samples actions around the network's mean (std = exp(log_std)); evaluation drives the
mean itself. This script compares the two for each checkpoint given:
  - mean policy: the deterministic evaluation episode (as ``evaluate`` drives it),
  - noisy policy: 32 cars, one episode each from the start line, actions sampled exactly as
    during training (the sim clips them to [-1, 1]).
For each it prints laps, how episodes ended and the average *applied* throttle (after clipping)
over all decisions, plus the top speed. Clipping a noisy throttle near +1 lowers its average:
with mean 0.9 and std 0.6, half the samples land above 1 and get cut to 1, while the low half
keeps its full spread below.

Run: uv run python docs/lessons/scripts/stochastic_vs_mean.py runs/reinforce-<timestamp>-s0 25 latest
"""

import json
import sys

import numpy as np
import torch

from racer.config import EnvConfig, from_dict
from racer.policy import load_policy
from racer.sim import BatchSim
from racer.track import load_track


def drive(track, env, actor, n: int, noisy: bool, seed: int = 0) -> dict:
    sim = BatchSim(track, env, n_cars=n, seed=seed, auto_reset=False)
    obs = sim.reset()
    crashed = np.zeros(n, dtype=bool)
    throttle, top = [], np.zeros(n)
    while sim.alive.any():
        alive = sim.alive.copy()
        with torch.no_grad():
            dist = actor.dist(torch.as_tensor(obs, dtype=torch.float32))
            action = (dist.sample() if noisy else dist.mean).numpy()
        applied = np.clip(action, -1.0, 1.0)
        throttle.extend(applied[alive, 1])
        result = sim.step(action)
        crashed |= result.terminated
        top = np.maximum(top, np.where(alive, sim.state.speed, 0.0))
        obs = result.obs
    return {"laps": sim.laps.copy(), "crashed": crashed, "throttle": float(np.mean(throttle)),
            "top": top, "progress": sim.traveled.copy()}


def main() -> None:
    run, which = sys.argv[1], sys.argv[2:]
    config = json.loads(open(f"{run}/config.json").read())
    env = from_dict(EnvConfig, config["env"])
    track = load_track(config["train"]["track"])
    torch.set_num_threads(2)
    for w in which:
        name = "latest.pt" if w == "latest" else f"ckpt_{int(w):05d}.pt"
        actor = load_policy(f"{run}/checkpoints/{name}").actor
        torch.manual_seed(0)
        det = drive(track, env, actor, 1, noisy=False)
        noisy = drive(track, env, actor, 32, noisy=True)
        std = actor.log_std.detach().exp().numpy()
        laps = np.bincount(noisy["laps"], minlength=3)
        print(f"{name} (std steer {std[0]:.2f}, throttle {std[1]:.2f})")
        print(f"  mean policy : {int(det['laps'][0])} laps, "
              f"{'crashed' if det['crashed'][0] else 'no crash'} after {det['progress'][0]:.0f} m, "
              f"avg applied throttle {det['throttle']:+.2f}, top speed {det['top'][0]:.1f} m/s")
        print(f"  noisy policy: 32 episodes, laps 0/1/2 = {laps[0]}/{laps[1]}/{laps[2]}, "
              f"crashed {int(noisy['crashed'].sum())}, avg applied throttle {noisy['throttle']:+.2f}, "
              f"median top speed {np.median(noisy['top']):.1f} m/s")


if __name__ == "__main__":
    main()
