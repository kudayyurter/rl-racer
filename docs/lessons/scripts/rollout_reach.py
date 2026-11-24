"""Lesson 09: how much of each training track does a random-track run actually drive?

In random-track mode ``train_ppo`` moves every car to a new track every ``track_every``
rollouts with a full ``reset()``, so all cars restart at the start line, standing still. With
``track_every = 1`` a track is driven for one rollout only: ``n_steps`` decisions, 128 x 4/60 s
= 8.53 s. This script measures what that window covers: for each checkpoint given, it drives
one car with the deterministic policy from the start line of the run's first training tracks
(the same seed sequence the run drew) for ``n_steps * track_every`` decisions, respawning after
an ending as training does, and reports the furthest point along the track it reached and what
share of the lap that is.

The deterministic car stands in for the 16 noisy training cars. A noisy car that crashes
earlier covers less, so read the numbers as roughly an upper bound on what training sees. The
distance is per episode, so after a respawn it starts again from 0; more than one lap counts
as 100%.

Arguments: ``RUN`` (checkpoints/latest.pt) or ``RUN@ITERATION``.

Run: uv run python docs/lessons/scripts/rollout_reach.py runs/ppo-<ts>-s0 runs/ppo-<ts>-s0@100
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from racer.config import EnvConfig, from_dict
from racer.policy import policy_from_checkpoint
from racer.sim import BatchSim
from racer.trackgen import track_for_seed, training_seed

N_TRACKS = 20          # first training tracks of the run to measure on
N_HELDOUT = 10         # held-out seeds the run's training seeds skip (as in ppo.py)


def main() -> None:
    for arg in sys.argv[1:]:
        run, _, iteration = arg.partition("@")
        name = f"ckpt_{int(iteration):05d}.pt" if iteration else "latest.pt"
        path = Path(run) / "checkpoints" / name
        ckpt = torch.load(path, weights_only=False)
        policy = policy_from_checkpoint(ckpt, path)
        config = json.loads((Path(run) / "config.json").read_text())
        train, cfg = config["train"], from_dict(EnvConfig, config["env"])
        window = train["n_steps"] * train["track_every"]
        rng = np.random.default_rng(train["seed"])   # the run's own track_rng
        reach, share = [], []
        for _ in range(N_TRACKS):
            track = track_for_seed(training_seed(rng, N_HELDOUT))
            sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=True)
            obs, furthest = sim.reset(), 0.0
            for _ in range(window):
                before = float(sim.traveled[0])
                res = sim.step(policy.act(obs))
                obs = res.obs
                # After an ending the car respawns and traveled restarts from 0: use final value.
                done = bool(res.terminated[0] or res.truncated[0])
                furthest = max(furthest, before if done else float(sim.traveled[0]))
            reach.append(furthest)
            share.append(min(furthest / track.length, 1.0))
        seconds = window * cfg.dt * cfg.action_repeat
        print(f"{arg} (iteration {ckpt['iteration']}): window {window} decisions = {seconds:.2f} s "
              f"per track; furthest point reached on {N_TRACKS} training tracks: median "
              f"{np.median(reach):.0f} m ({min(reach):.0f} .. {max(reach):.0f}), "
              f"median {100 * np.median(share):.0f}% of a lap")


if __name__ == "__main__":
    main()
