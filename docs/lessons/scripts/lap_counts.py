"""Lesson 07: how many evaluations drove 0, 1 and 2 laps, in each quarter of a run.

A run that keeps "lapping" can still change character: on twisty two laps take about 84-90 s at
a 42-45 s lap, so a driver that slows to 48 s a lap hits the 90 s time limit after one lap and its
evaluation reads 1 lap with no crash. ``eval_history.py`` counts any lap as a success; this
script splits the evaluations of each run into quarters of its iterations and counts 0-, 1- and
2-lap evaluations in each, plus the median best lap of the lapping ones.

Run: uv run python docs/lessons/scripts/lap_counts.py runs/ppo-<timestamp>-s0 ...
"""

import math
import sys
from pathlib import Path

import numpy as np

from racer.runlog import RunDir


def main() -> None:
    for arg in sys.argv[1:]:
        rows = RunDir.open(Path(arg)).read_metrics()
        evals = [r for r in rows if not math.isnan(r["eval_laps"])]
        last = int(rows[-1]["iteration"])
        parts = []
        for q in range(4):
            lo, hi = q * last / 4, (q + 1) * last / 4
            chunk = [r for r in evals if lo < r["iteration"] <= hi]
            laps = np.bincount([int(r["eval_laps"]) for r in chunk], minlength=3)
            times = [r["eval_best_lap_s"] for r in chunk if r["eval_laps"] > 0]
            med = f"{np.median(times):5.2f} s" if times else "   -   "
            parts.append(f"Q{q + 1} {laps[0]:3d}/{laps[1]:3d}/{laps[2]:3d} lap {med}")
        print(f"{Path(arg).name}: evaluations with 0/1/2 laps, median best lap, per quarter")
        print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
