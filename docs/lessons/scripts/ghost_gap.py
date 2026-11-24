"""Lesson 04: why ghosts lead the live car during its first lap.

Drives the newest checkpoint of a run for one evaluation episode (the same deterministic
episode racer/evaluate.py drives), prints each lap's time, and compares where the car is
``--at`` seconds into lap 1 (a standing start) with where it is the same number of seconds into
its fastest lap (the lap that becomes its ghost). The difference is how far the ghost of the
very same network is ahead on screen at that moment of the live car's first lap.

Run: uv run python docs/lessons/scripts/ghost_gap.py runs/evolution-<timestamp>-s<seed> [--at 15]
"""

import argparse
import json

import numpy as np

from racer.config import EnvConfig, from_dict
from racer.policy import load_policy
from racer.sim import BatchSim
from racer.track import load_track


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--at", type=float, default=15.0, help="seconds into the lap")
    args = parser.parse_args()

    config = json.loads(open(f"{args.run}/config.json").read())
    env = from_dict(EnvConfig, config["env"])
    track = load_track(config["train"]["track"])
    policy = load_policy(f"{args.run}/checkpoints/latest.pt")
    sim = BatchSim(track, env, n_cars=1, auto_reset=False)
    obs = sim.reset()

    times, traveled, laps = [0.0], [0.0], []   # laps: (episode time at the line, lap time)
    while sim.alive[0]:
        result = sim.step(policy.act(obs))
        obs = result.obs
        times.append(float(sim.episode_time[0]))
        traveled.append(float(sim.traveled[0]))
        if result.lap_completed[0]:
            laps.append((times[-1], float(result.lap_time[0])))

    for k, (_, lap_time) in enumerate(laps, start=1):
        print(f"lap {k}: {lap_time:.2f} s" + ("  (standing start)" if k == 1 else "  (flying)"))
    if not laps:
        print("no lap completed")
        return
    best = min(range(len(laps)), key=lambda k: laps[k][1])
    # The best lap began when the car crossed the line, i.e. after the earlier laps' times
    # (more exact than the episode time of the decision in which the crossing happened).
    lap_start_time = sum(lap_time for _, lap_time in laps[:best])
    lap_start_dist = best * track.length
    at_lap1 = np.interp(args.at, times, traveled)
    at_best = np.interp(lap_start_time + args.at, times, traveled) - lap_start_dist
    print(f"{args.at:g} s into lap 1: {at_lap1:.1f} m;  {args.at:g} s into lap {best + 1} "
          f"(the ghost): {at_best:.1f} m;  ghost ahead by {at_best - at_lap1:.1f} m")


if __name__ == "__main__":
    main()
