"""Lesson 01 experiment: what would a global nearest-point search do at a hairpin?

Uses the same U-turn track as tests/test_track.py (two straights 14 m apart, road 12 m wide, so
the left wall of the near straight (y = 6) and the right wall of the far one (y = 8) are only 2 m
apart). Points along the near straight are projected twice:

* windowed: Track.project with a hint at the right place and half = 10 (what BatchSim does)
* global:   the same call with half = M // 2, i.e. every sample on the track is a candidate

and the script prints the progress s each one reports, plus the progress reward the difference
would pay in one step (progress_scale * jump). Rows: on the centerline (y = 0), near the left
wall (y = 5), and in the 2 m grass gap between the straights (y = 8, the point the test uses).

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
    scale = EnvConfig().progress_scale
    print(f"hairpin track: length {t.length:.1f} m, {m} samples, spacing {t.spacing:.2f} m")
    print(f"{'x':>5} {'y':>4} {'windowed s':>11} {'global s':>9} {'jump':>8} {'reward':>7}")
    for y in (0.0, 5.0, 8.0):
        for x in (40.0, 60.0, 80.0):
            p = np.array([[x, y]])
            hint = np.array([int(round(x / t.spacing))])   # the car's own stretch of road
            _, s_win = t.project(p, hint)
            _, s_glob = t.project(p, hint, half=m // 2)
            jump = s_glob[0] - s_win[0]
            print(f"{x:5.0f} {y:4.0f} {s_win[0]:11.1f} {s_glob[0]:9.1f} {jump:+8.1f} "
                  f"{scale * jump:+7.1f}")

    twisty = load_track("twisty")
    mt = len(twisty.center)
    print(f"\nproject() for 256 cars on twisty ({mt} samples):")
    print(f"  windowed (half = 10, 21 candidates): {time_project(twisty, 10):8.0f} us per call")
    print(f"  global (half = {mt // 2}, {2 * (mt // 2) + 1} candidates): "
          f"{time_project(twisty, mt // 2):8.0f} us per call")


if __name__ == "__main__":
    main()
