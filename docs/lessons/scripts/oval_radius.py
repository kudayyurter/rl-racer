"""Lesson 02 measurement: the oval's real corner radius, and how a 24 m/s car still gets round.

The oval's end-turn control points sit on 50 m circles, but the spline through them is not a
perfect circle. This script prints:

1. the radius of the sampled centerline through the body of the end turns (three-point circles
   6 m apart, only where the radius is under 60 m), its minimum and maximum, and how many meters
   of the lap are tighter than 45 m;
2. a lap with the target-speed driver from grip_laps.py at the fastest target that survived
   there, 24 m/s at grip 12 and 40 m/s at grip 30: the car's speed through the turns, the tightest radius its own path
   actually drove (speed / yaw rate per decision), and its largest offset from the centerline.

Run: uv run python docs/lessons/scripts/oval_radius.py   (a few seconds)
"""

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for tests.helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))       # this folder, for grip_laps

from grip_laps import TargetSpeed  # noqa: E402
from racer.config import EnvConfig  # noqa: E402
from racer.sim import BatchSim, wrap_angle  # noqa: E402
from racer.track import load_track  # noqa: E402

TURN_RADIUS_M = 100.0   # centerline points tighter than this count as "in a turn"
BODY_RADIUS_M = 60.0    # tighter than this: the body of a turn, past its entry and exit
LAPS = ((12.0, 24.0), (30.0, 40.0))   # (max_lat_accel, target speed): the fastest clean laps in grip_laps.py


def centerline_radius(track, span_m: float = 6.0) -> np.ndarray:
    k = max(1, round(span_m / track.spacing))
    p0, p1, p2 = (np.roll(track.center, -s, axis=0) for s in (-k, 0, k))
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    twice_area = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    with np.errstate(divide="ignore"):
        return a * b * c / (2.0 * twice_area)


def main() -> None:
    track = load_track("oval")
    cfg = EnvConfig()
    radius = centerline_radius(track)
    body = radius < BODY_RADIUS_M
    print(f"centerline radius in the body of the turns: min {radius[body].min():.1f} m, "
          f"max {radius[body].max():.1f} m; tighter than 45 m for "
          f"{(radius < 45.0).sum() * track.spacing:.0f} m of the {track.length:.0f} m lap")
    print(f"formula speed at the tightest point: sqrt({cfg.car.max_lat_accel} * "
          f"{radius.min():.1f}) = {math.sqrt(cfg.car.max_lat_accel * radius.min()):.1f} m/s")

    for grip, target in LAPS:
        drive_lap(track, cfg, radius, grip, target)


def drive_lap(track, base_cfg, radius, grip: float, target: float) -> None:
    """One episode with the target-speed driver; print speed, path radius and offset in turns."""
    cfg = dataclasses.replace(base_cfg, car=dataclasses.replace(base_cfg.car, max_lat_accel=grip))
    sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
    policy = TargetSpeed(cfg, track, target)
    obs = sim.reset()
    decision_s = cfg.dt * cfg.action_repeat
    speeds, path_r, offsets, laps, crashed = [], [], [], 0, False
    while sim.alive[0]:
        heading_before = sim.state.heading[0]
        res = sim.step(policy.act(obs))
        obs = res.obs
        laps += int(res.lap_completed[0])
        crashed |= bool(res.terminated[0])
        i = int(np.argmin(((track.center - sim.state.pos[0]) ** 2).sum(axis=1)))
        if radius[i] < TURN_RADIUS_M and sim.episode_time[0] > 10.0:   # turns, after the start
            turn_rate = abs(wrap_angle(np.array([sim.state.heading[0] - heading_before]))[0])
            speeds.append(sim.state.speed[0])
            path_r.append(sim.state.speed[0] * decision_s / max(turn_rate, 1e-9))
            offsets.append(float((sim.state.pos[0] - track.center[i]) @ track.normal[i]))
    print(f"\n{target:.0f} m/s target, grip {grip:.0f}: laps {laps}, crashed {crashed}")
    print(f"  speed in the turns: {min(speeds):.1f} to {max(speeds):.1f} m/s")
    print(f"  tightest radius the car drove: {min(path_r):.1f} m "
          f"(the grip limit allows {max(speeds) ** 2 / grip:.1f} m at {max(speeds):.1f} m/s)")
    print(f"  offset from the centerline in the turns: {min(offsets):+.2f} to {max(offsets):+.2f} m "
          f"(+ = left/inside; walls at +-{track.width / 2:.0f} m)")

if __name__ == "__main__":
    main()
