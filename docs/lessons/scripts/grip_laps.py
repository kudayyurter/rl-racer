"""Lesson 02 experiment: best scripted lap with max_lat_accel 12 vs 30.

A stand-in for "drive it yourself and compare your best lap": the scripted P-controller from
tests/helpers.py steers (from the observation only), and the throttle holds one target speed for
the whole lap (proportional throttle/brake toward the target). For each grip limit we sweep the
target speed and keep the fastest lap that finishes without crashing. The controller never
brakes for a corner, so the speed it can hold is set by the tightest corner on the track.

Each episode is a normal episode with the default env config: start from rest on the start line,
end at 2 laps (max_laps), a crash or wrong way, or the 90 s time limit. "best lap" is the faster
of the laps driven (the second lap has a flying start). In the per-target rows, "crash" means the
car hit a wall and "slow" means the 90 s ran out before a single lap.

Run: uv run python docs/lessons/scripts/grip_laps.py   (about 30 s)
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for tests.helpers

from racer.config import EnvConfig  # noqa: E402
from racer.sim import BatchSim  # noqa: E402
from racer.track import load_track  # noqa: E402
from tests.helpers import ScriptedOvalPolicy  # noqa: E402

GRIPS = (12.0, 30.0)
TARGETS = range(10, 47, 2)    # target speeds to try, m/s


class TargetSpeed(ScriptedOvalPolicy):
    """The helper's steering, with throttle/brake proportional to (target - speed)."""

    def __init__(self, cfg, track, target: float):
        super().__init__(cfg, track)
        self.target = target

    def act(self, obs):
        action = super().act(obs)
        speed = obs[:, -3] * self.max_speed
        action[:, 1] = np.clip(0.5 * (self.target - speed), -1.0, 1.0)
        return action


def run_episode(policy, track, cfg) -> tuple[str, list[float]]:
    """Drive one episode; return how it ended ("crash", "slow" or "laps") and the lap times."""
    sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
    obs = sim.reset()
    lap_times = []
    while sim.alive[0]:
        res = sim.step(policy.act(obs))
        obs = res.obs
        if res.lap_completed[0]:
            lap_times.append(float(res.lap_time[0]))
        if res.terminated[0]:
            return "crash", lap_times
    return ("laps" if lap_times else "slow"), lap_times


def main() -> None:
    base = EnvConfig()
    for track_name in ("oval", "twisty"):
        track = load_track(track_name)
        for grip in GRIPS:
            cfg = dataclasses.replace(base, car=dataclasses.replace(base.car, max_lat_accel=grip))
            best = None
            rows = []
            for target in TARGETS:
                end, laps = run_episode(TargetSpeed(cfg, track, target), track, cfg)
                if end == "laps":
                    rows.append(f"{target}:{min(laps):.2f}")
                    if best is None or min(laps) < best[1]:
                        best = (target, min(laps), len(laps))
                else:
                    # A crash after a lap still counts as a crash: the run was not clean.
                    rows.append(f"{target}:{end}")
            print(f"{track_name:6} max_lat_accel={grip:4.0f}  "
                  + (f"best lap {best[1]:.2f} s at target {best[0]} m/s ({best[2]} laps)"
                     if best else "no clean lap at any target"))
            print("    target:lap  " + "  ".join(rows))


if __name__ == "__main__":
    main()
