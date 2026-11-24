"""Lesson 04: how often does an evaluation reproduce the training fitness of the same genome?

For every evaluation row in the given evolution runs, compares ``best_fitness`` (the genome's
return in training, through ``forward_population``) with ``eval_return`` (the same genome driven
by ``racer.evaluate``, through ``EvolutionPolicy.act``).

Counting rule: an elite that survives a plateau is evaluated again every fifth generation, so
rows repeat genomes. Each genome is counted once per run directory: two rows of the same run are
the same genome when the weights in their checkpoints (``checkpoints/ckpt_XXXXX.pt``) are
identical. Runs are not merged with each other, even when one run repeats another's first
generations. A genome's difference is the largest |best_fitness - eval_return| over its rows.

Run: uv run python docs/lessons/scripts/eval_vs_fitness.py runs/evolution-<a> runs/evolution-<b> ...
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

from racer.runlog import RunDir


def genome_differences(run: Path) -> list[float]:
    """Largest |training fitness - evaluation return| for each distinct evaluated genome."""
    by_genome: dict[bytes, float] = {}
    for row in RunDir.open(run).read_metrics():
        if math.isnan(row["eval_return"]):
            continue
        ckpt = torch.load(run / "checkpoints" / f"ckpt_{int(row['iteration']):05d}.pt",
                          weights_only=False)
        key = np.asarray(ckpt["weights"], dtype=float).tobytes()
        diff = abs(row["best_fitness"] - row["eval_return"])
        by_genome[key] = max(by_genome.get(key, 0.0), diff)
    return list(by_genome.values())


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]]
    diffs, rows = [], []
    for run in runs:
        d = genome_differences(run)
        diffs += d
        rows += [(run.name, x) for x in d if x > 0.1]
    diffs = np.array(diffs)
    print(f"{len(runs)} runs, {len(diffs)} distinct evaluated genomes (counted per run)")
    print(f"  |training - evaluation| <= 0.005: {int((diffs <= 0.005).sum())}")
    print(f"  |training - evaluation| <= 0.1:   {int((diffs <= 0.1).sum())}")
    for name, x in sorted(rows, key=lambda r: r[1]):
        print(f"  > 0.1: {x:6.2f}  {name}")


if __name__ == "__main__":
    main()
