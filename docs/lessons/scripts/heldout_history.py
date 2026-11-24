"""Lesson 09: held-out and twisty evaluations over a random-track run, from metrics.csv.

Random-track PPO (``tracks = "random"``) logs two evaluations every ``eval_every`` iterations:
``eval_laps`` on the eval track (twisty) and ``eval_heldout_laps_mean``, the mean laps of one
deterministic episode on each of the 10 held-out tracks (0, 1 or 2 laps each, so the mean runs
0..2 and 2.00 means every held-out track was driven both laps). For each run this prints:

  * the first evaluation where the held-out mean reached 1.0 and 2.0, and where twisty lapped;
  * per quarter of the run's evaluations: the median held-out mean, how many evaluations had a
    perfect 2.00, and twisty evaluations with 0/1/2 laps plus the median best twisty lap.

``--until STEPS`` keeps only evaluations at or before STEPS env steps, to compare the start of a
long run with a shorter run of the same config and seed (PPO does not depend on its budget
until it stops, so the two log the same rows up to there).

Run: uv run python docs/lessons/scripts/heldout_history.py runs/ppo-<ts>-s0 runs/ppo-<ts>-s1
     uv run python docs/lessons/scripts/heldout_history.py --until 1001472 runs/ppo-<ts>-s1
"""

import math
import statistics
import sys

from racer.runlog import RunDir


def first(evals: list[dict], test) -> str:
    row = next((r for r in evals if test(r)), None)
    return "never" if row is None else f"iteration {int(row['iteration'])} ({int(row['env_steps']):,} steps)"


def main() -> None:
    args, until = sys.argv[1:], math.inf
    if args[:1] == ["--until"]:
        until, args = int(args[1]), args[2:]
    for run in args:
        evals = [r for r in RunDir.open(run).read_metrics()
                 if not math.isnan(r["eval_laps"]) and r["env_steps"] <= until]
        print(f"{run}: {len(evals)} evaluations" + (f" up to {until:,} steps" if until < math.inf else ""))
        print(f"  held-out mean >= 1.0: {first(evals, lambda r: r['eval_heldout_laps_mean'] >= 1.0)}")
        print(f"  held-out mean = 2.0:  {first(evals, lambda r: r['eval_heldout_laps_mean'] >= 2.0)}")
        print(f"  twisty lapped:        {first(evals, lambda r: r['eval_laps'] >= 1)}")
        n = len(evals)
        for q in range(4):
            part = evals[q * n // 4:(q + 1) * n // 4]
            heldout = [r["eval_heldout_laps_mean"] for r in part]
            perfect = sum(h >= 2.0 for h in heldout)
            laps = [sum(int(r["eval_laps"]) == k for r in part) for k in (0, 1, 2)]
            lap_times = [r["eval_best_lap_s"] for r in part if not math.isnan(r["eval_best_lap_s"])]
            best = f"{statistics.median(lap_times):.2f} s" if lap_times else "-"
            print(f"  Q{q + 1} (iterations {int(part[0]['iteration'])}-{int(part[-1]['iteration'])}): "
                  f"held-out median {statistics.median(heldout):.2f}, {perfect:3d}/{len(part)} at 2.00 | "
                  f"twisty 0/1/2 laps {laps[0]:3d}/{laps[1]:3d}/{laps[2]:3d}, median lap {best}")


if __name__ == "__main__":
    main()
