"""Lesson 02 measurement: why does a constant 0.4 throttle crash on the oval's first turn?

Drives one car on the oval with the scripted P-controller from tests/helpers.py (it steers from
the observation only, like a learned policy). Two versions:

* "coast above 20 m/s": the helper as written (throttle 0.4, but 0 above 20 m/s)
* "constant 0.4":       the same steering, throttle held at 0.4 the whole time

For each it prints how the episode ended, the speed when the car reaches the end of the first
straight (x = 100 m, where the first turn begins), the top speed, and the car's sideways offset
from the centerline at the end. It also prints the oval's nominal corner radius (the 50 m
circle its end-turn control points sit on; oval_radius.py measures the spline's real radius,
38.9 to 57.5 m) and the fastest speed the grip limit allows there, v = sqrt(max_lat_accel * R).

Run: uv run python docs/lessons/scripts/constant_throttle.py   (a few seconds)
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for tests.helpers

from racer.config import EnvConfig  # noqa: E402
from racer.sim import BatchSim  # noqa: E402
from racer.track import load_track  # noqa: E402
from tests.helpers import ScriptedOvalPolicy  # noqa: E402

TURN_START_X = 100.0   # the oval's straights end at x = +-100 m; the end turns have radius 50 m
NOMINAL_RADIUS_M = 50.0   # the circle the end-turn control points sit on (tracks/oval.json)


class ConstantThrottle(ScriptedOvalPolicy):
    """The helper's steering, but the throttle never coasts."""

    def act(self, obs):
        action = super().act(obs)
        action[:, 1] = 0.4
        return action


def drive(policy, track, cfg) -> dict:
    sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
    obs = sim.reset()
    entry_speed, top_speed, laps, lap_times = None, 0.0, 0, []
    while sim.alive[0]:
        res = sim.step(policy.act(obs))
        obs = res.obs
        speed = float(sim.state.speed[0])
        top_speed = max(top_speed, speed)
        if entry_speed is None and sim.state.pos[0, 0] >= TURN_START_X:
            entry_speed = speed
        if res.lap_completed[0]:
            lap_times.append(float(res.lap_time[0]))
        if res.terminated[0] or res.truncated[0]:
            end = "crashed" if res.terminated[0] else "finished (truncated)"
            laps = int(sim.laps[0])
    # Sideways offset at the end, from the nearest centerline sample (+ = left of center).
    i = int(np.argmin(((track.center - sim.state.pos[0]) ** 2).sum(axis=1)))
    offset = float((sim.state.pos[0] - track.center[i]) @ track.normal[i])
    return {"end": end, "laps": laps, "lap_times": lap_times, "entry_speed": entry_speed,
            "top_speed": top_speed, "end_offset": offset, "end_pos": sim.state.pos[0].round(1),
            "time": float(sim.episode_time[0])}


def main() -> None:
    track = load_track("oval")
    cfg = EnvConfig()
    radius = NOMINAL_RADIUS_M
    print(f"oval: width {track.width} m (walls {track.width / 2} m either side), "
          f"nominal corner radius {radius} m")
    print(f"grip limit {cfg.car.max_lat_accel} m/s^2 -> fastest speed round a {radius} m corner: "
          f"sqrt({cfg.car.max_lat_accel} * {radius}) = {math.sqrt(cfg.car.max_lat_accel * radius):.1f} m/s")
    for label, policy in (("coast above 20 m/s", ScriptedOvalPolicy(cfg, track)),
                          ("constant 0.4", ConstantThrottle(cfg, track))):
        r = drive(policy, track, cfg)
        laps = ", ".join(f"{t:.2f} s" for t in r["lap_times"]) or "none"
        print(f"\n{label}:")
        print(f"  ended: {r['end']} after {r['time']:.2f} s, laps {r['laps']} ({laps})")
        print(f"  speed at the end of the first straight (x = {TURN_START_X:.0f} m): "
              f"{r['entry_speed']:.1f} m/s, top speed {r['top_speed']:.1f} m/s")
        print(f"  final position {r['end_pos']}, {r['end_offset']:+.2f} m from the centerline")
        tightest = r["top_speed"] ** 2 / cfg.car.max_lat_accel
        print(f"  at {r['top_speed']:.1f} m/s the tightest turn the grip allows has radius "
              f"v^2 / a = {tightest:.1f} m")


if __name__ == "__main__":
    main()
