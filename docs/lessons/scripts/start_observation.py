"""Lesson 03: print the observation a car gets on the start line of twisty, entry by entry.

Shows the 12 numbers a policy receives (9 rays, speed, steer, heading error) with their raw
values and the scaled values that actually go into the observation. Change the track or the
env config below to see how the vector changes.

Run: uv run python docs/lessons/scripts/start_observation.py
"""

import math

import numpy as np

from racer.config import EnvConfig
from racer.sim import BatchSim
from racer.track import load_track


def main() -> None:
    cfg = EnvConfig()
    sim = BatchSim(load_track("twisty"), cfg, n_cars=1)
    obs = sim.reset()[0]
    angles = np.degrees(np.linspace(-math.radians(cfg.ray_spread_deg) / 2,
                                    math.radians(cfg.ray_spread_deg) / 2, cfg.n_rays))
    print(f"{'entry':>5}  {'meaning':28} {'raw':>10} {'obs':>7}")
    for i, (a, dist) in enumerate(zip(angles, sim.last_rays[0])):
        print(f"{i:>5}  {f'ray at {a:+.1f} deg':28} {dist:8.2f} m {obs[i]:7.3f}")
    n = cfg.n_rays
    print(f"{n:>5}  {'speed / max_speed':28} {sim.state.speed[0]:6.2f} m/s {obs[n]:7.3f}")
    print(f"{n + 1:>5}  {'steer / max_steer':28} {sim.state.steer[0]:6.2f} rad {obs[n + 1]:7.3f}")
    print(f"{n + 2:>5}  {'heading error / pi':28} {obs[n + 2] * math.pi:6.2f} rad {obs[n + 2]:7.3f}")


if __name__ == "__main__":
    main()
