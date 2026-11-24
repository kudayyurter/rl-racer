"""Lessons 05-06: summary statistics of one metrics.csv column over an iteration window.

Reads every row (not a sample) of a run's metrics.csv with ``first <= iteration <= last`` and
prints, for the column given: how many rows had a value, the mean, median, min and max, and
optionally how many rows exceeded a threshold. Rows where the column is NaN (an A2C rollout in
which no episode ended has no episode length) are skipped and counted separately.

Run: uv run python docs/lessons/scripts/metric_window.py runs/a2c-<timestamp>-s2 value_loss 8610 8699 15
     (run, column, first iteration, last iteration, optional threshold)
"""

import sys

import numpy as np

from racer.runlog import RunDir


def main() -> None:
    run, column, first, last = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else None
    rows = [r for r in RunDir.open(run).read_metrics() if first <= r["iteration"] <= last]
    values = np.array([r[column] for r in rows], dtype=float)
    ok = values[~np.isnan(values)]
    line = (f"{run} {column}, iterations {first}-{last}: {len(ok)} rows with a value "
            f"({len(values) - len(ok)} NaN); mean {ok.mean():.3f}, median {np.median(ok):.3f}, "
            f"min {ok.min():.3f}, max {ok.max():.3f}")
    if threshold is not None:
        line += f"; above {threshold:g}: {int((ok > threshold).sum())} of {len(ok)}"
    print(line)


if __name__ == "__main__":
    main()
