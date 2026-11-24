"""Lesson 08: compare the fastest recorded laps (ghosts) of several runs: line and speed.

A ghost is the pose trace (x, y, heading every decision, 1/15 s) of a run's fastest evaluation
lap. For each run given, this script takes its fastest ghost (or the ghost of RUN@ITERATION)
and reports, from the poses alone:
  - lap time, average and top speed, and the slowest speed of the lap (with where it happened);
  - the racing line: how far the car runs from the centerline (mean |offset|, and the share of
    the lap spent more than 3 m off it; the road is 12 m wide, the car 2 m, so the walls are at
    about +-5 m for the car's center);
  - the time spent in each eighth of the track (141 m each on twisty), to show where one lap
    gains on another.

Run: uv run python docs/lessons/scripts/line_compare.py runs/ppo-<ts>-s2 runs/ppo-<ts>-s2@80 ...
"""

import json
import sys
from pathlib import Path

import numpy as np

from racer.track import load_track

SECTIONS = 8


def load_ghost(arg: str):
    run, _, iteration = arg.partition("@")
    files = sorted((Path(run) / "ghosts").glob("ghost_*.npz"))
    if iteration:
        files = [f for f in files if f.name == f"ghost_{int(iteration):05d}.npz"]
    ghosts = []
    for f in files:
        with np.load(f) as z:
            ghosts.append((float(z["lap_time"]), int(z["iteration"]), z["t"], z["poses"]))
    if not ghosts:
        raise SystemExit(f"no ghost for {arg}")
    track = json.loads((Path(run) / "config.json").read_text())["train"]["track"]
    return min(ghosts, key=lambda g: g[0]), track


def main() -> None:
    print("time per eighth of the lap (s), from the start line")
    for arg in sys.argv[1:]:
        (lap, iteration, t, poses), track_name = load_ghost(arg)
        track = load_track(track_name)
        pos = poses[:, :2]
        # Project every pose, carrying the index forward (the same windowed search the sim uses);
        # the first pose is placed by a global nearest-sample search.
        idx = np.array([int(((track.center - pos[0]) ** 2).sum(axis=1).argmin())])
        offs, s_all = [], []
        for p in pos:
            idx, s = track.project(p[None, :], idx, half=10)
            offs.append(float(np.dot(p - track.center[idx[0]], track.normal[idx[0]])))
            s_all.append(float(s[0]))
        offs = np.array(offs)
        # Distance along the track since the lap began (unwrapped), and speed from the poses.
        s_lap = np.unwrap(np.array(s_all) * 2 * np.pi / track.length) * track.length / (2 * np.pi)
        s_lap -= s_lap[0]
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
        slow = int(speed.argmin())
        edges = np.linspace(0, track.length, SECTIONS + 1)
        cross = np.interp(edges, s_lap, t)  # time at which each section boundary is passed
        sections = " ".join(f"{d:5.2f}" for d in np.diff(cross))
        print(f"{arg} (iteration {iteration}): lap {lap:.2f} s, average {track.length / lap:.1f} m/s, "
              f"top {speed.max():.1f} m/s, slowest {speed.min():.1f} m/s at {s_lap[slow]:.0f} m")
        print(f"    line: mean |offset| {np.abs(offs).mean():.2f} m, more than 3 m off "
              f"{100 * (np.abs(offs) > 3).mean():.0f}% of the lap, max |offset| "
              f"{np.abs(offs).max():.2f} m")
        print(f"    sections: {sections}")


if __name__ == "__main__":
    main()
