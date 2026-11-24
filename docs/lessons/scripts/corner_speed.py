"""Lesson 09: how fast does a driver arrive at a track's tightest corner?

Lesson 02's braking problem: the grip limit allows a speed of about sqrt(max_lat_accel * r)
through a corner of radius r (12 m/s for r = 12 m), so a car must brake *before* the corner,
which it can only learn by having arrived too fast and crashed there. This script finds the
track's tightest corner (the sample with the smallest three-point radius, as ``track_is_valid``
measures it) and, for each checkpoint, drives one deterministic episode from the start line
(the run's own env config) and prints the speed 60 m and 20 m before that corner, the lowest
speed within 20 m either side of it, and how the episode ended and where. Only the first pass
of the corner is reported (lap 1).

Track: ``twisty`` or a held-out seed such as ``10009`` (the same track ``heldout_tracks``
builds). Checkpoints: ``RUN``, ``RUN@ITERATION`` or ``RUN@best`` as in ``heldout_eval.py``.

Run: uv run python docs/lessons/scripts/corner_speed.py twisty runs/ppo-<ts>-s0 runs/ppo-<ts>-s0@best
"""

import json
import math
import sys

import numpy as np
import torch

from heldout_eval import checkpoint_path   # same directory; sys.path[0] is this script's folder
from racer.config import EnvConfig, from_dict
from racer.policy import policy_from_checkpoint
from racer.sim import BatchSim
from racer.track import Track, load_track
from racer.trackgen import track_for_seed

CHECK_BEFORE_M = (60.0, 20.0)   # report the speed this far before the apex
APEX_WINDOW_M = 20.0            # lowest speed within this distance of the apex


def tightest_corner(track: Track, k: int = 3) -> tuple[float, float]:
    """(distance from the start line along the lap, radius) of the tightest corner."""
    p0, p1, p2 = (np.roll(track.center, -s, axis=0) for s in (-k, 0, k))
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    twice_area = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    with np.errstate(divide="ignore"):
        radius = a * b * c / (2.0 * twice_area)
    i = int(np.argmin(radius))
    start = track.cum_len[track.start_index]
    return float((track.cum_len[i] - start) % track.length), float(radius[i])


def main() -> None:
    name, args = sys.argv[1], sys.argv[2:]
    track = load_track("twisty") if name == "twisty" else track_for_seed(int(name), f"heldout_{name}")
    apex, radius = tightest_corner(track)
    for arg in args:
        run, path = checkpoint_path(arg)
        ckpt = torch.load(path, weights_only=False)
        policy = policy_from_checkpoint(ckpt, path)
        cfg = from_dict(EnvConfig, json.loads((run / "config.json").read_text())["env"])
        if arg is args[0]:
            print(f"{track.name}: tightest corner {radius:.1f} m radius at {apex:.0f} m from the "
                  f"start; grip-limit speed sqrt({cfg.car.max_lat_accel:g} x r) = "
                  f"{math.sqrt(cfg.car.max_lat_accel * radius):.1f} m/s")
        sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
        obs = sim.reset()
        trace = []   # (meters driven, speed) after each decision
        end = "2 laps or time limit"
        while sim.alive[0]:
            res = sim.step(policy.act(obs))
            obs = res.obs
            trace.append((float(sim.traveled[0]), float(sim.state.speed[0])))
            if res.terminated[0]:
                end = "crash" if res.reward[0] < -cfg.crash_penalty / 2 else "wrong way"
        meters, speed = np.array(trace).T

        def speed_at(m: float) -> str:
            i = np.searchsorted(meters, m)
            return f"{speed[i]:5.1f}" if i < len(meters) else "    -"

        near = (meters > apex - APEX_WINDOW_M) & (meters < apex + APEX_WINDOW_M)
        lowest = f"{speed[near].min():5.1f}" if near.any() else "    -"
        before = ", ".join(f"{d:.0f} m before {speed_at(apex - d)}" for d in CHECK_BEFORE_M)
        print(f"  {arg} (iteration {ckpt['iteration']}): speed {before}, lowest near apex "
              f"{lowest} m/s; episode ended: {end} at {meters[-1]:.1f} m")


if __name__ == "__main__":
    main()
