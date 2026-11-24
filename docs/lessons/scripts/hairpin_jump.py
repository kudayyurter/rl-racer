"""Lesson 01 experiment: what would a global nearest-point search do at a hairpin?

Uses the same U-turn track as tests/test_track.py (two straights 14 m apart, road 12 m wide, so
the left wall of the near straight (y = 6) and the right wall of the far one (y = 8) are only 2 m
apart). Points along the near straight are projected twice:

* windowed: Track.project with a hint at the right place and half = 10 (what BatchSim does)
* global:   the same call with half = M // 2, i.e. every sample on the track is a candidate

For each point the script prints the progress s each search reports. Then it feeds the global
answer through BatchSim's own progress rule, as if the car had been at the windowed s one substep
earlier: ds is wrapped exactly as in racer/sim.py (ds = (s - s_prev + L/2) % L - L/2), so a jump of
more than half a lap reads as a large step *backwards*. It prints that ds, the progress reward it
would pay (progress_scale * ds), and whether it would end the episode by the wrong-way rule
(more than wrong_way_m backwards). Rows: on the centerline (y = 0), near the left wall (y = 5),
and at y = 8, the far road's edge (its right wall), which is nearer the far centerline (y = 14)
than the near one (y = 0). That is the point the test uses; the grass is 6 < y < 8.

Then it times both searches for 256 cars on twisty, since the window is also about cost.

Run: uv run python docs/lessons/scripts/hairpin_jump.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for tests.helpers

from racer.config import EnvConfig  # noqa: E402
from racer.track import build_track, load_track  # noqa: E402
from tests.helpers import hairpin_points  # noqa: E402


def time_project(track, half: int, n_cars: int = 256, repeats: int = 200) -> float:
    """Microseconds per project() call for n_cars cars spread around the track."""
    idx = np.linspace(0, len(track.center) - 1, n_cars).astype(int)
    pos = track.center[idx] + 0.5
    start = time.perf_counter()
    for _ in range(repeats):
        track.project(pos, idx, half=half)
    return (time.perf_counter() - start) / repeats * 1e6


def main() -> None:
    t = build_track("hairpin", hairpin_points(gap=14.0, straight=120.0), 12.0)
    m = len(t.center)
    cfg = EnvConfig()
    half_lap = t.length / 2
    print(f"hairpin track: length {t.length:.1f} m (half a lap {half_lap:.1f} m), {m} samples, "
          f"spacing {t.spacing:.2f} m")
    print(f"{'x':>5} {'y':>4} {'windowed s':>11} {'global s':>9} {'sim ds':>8} {'reward':>7}  outcome")
    for y in (0.0, 5.0, 8.0):
        for x in (40.0, 60.0, 80.0):
            p = np.array([[x, y]])
            hint = np.array([int(round(x / t.spacing))])   # the car's own stretch of road
            _, s_win = t.project(p, hint)
            _, s_glob = t.project(p, hint, half=m // 2)
            # The simulator's wrapped progress step (racer/sim.py), from the windowed s.
            ds = (s_glob[0] - s_win[0] + half_lap) % t.length - half_lap
            if abs(ds) < 1e-9:
                outcome = "no jump"
            elif -ds > cfg.wrong_way_m:
                outcome = f"fake wrong-way ending (> {cfg.wrong_way_m:.0f} m backwards)"
            else:
                outcome = "fake progress reward"
            print(f"{x:5.0f} {y:4.0f} {s_win[0]:11.1f} {s_glob[0]:9.1f} {ds:+8.1f} "
                  f"{cfg.progress_scale * ds:+7.1f}  {outcome}")

    twisty = load_track("twisty")
    mt = len(twisty.center)
    print(f"\nproject() for 256 cars on twisty ({mt} samples):")
    print(f"  windowed (half = 10, 21 candidates): {time_project(twisty, 10):8.0f} us per call")
    print(f"  global (half = {mt // 2}, {2 * (mt // 2) + 1} candidates): "
          f"{time_project(twisty, mt // 2):8.0f} us per call")


if __name__ == "__main__":
    main()
