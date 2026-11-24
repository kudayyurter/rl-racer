"""Lessons 04+: one line per run directory with the numbers the lessons quote from metrics.csv.

For each run: the method, track and the settings that differ between experiments, how many
iterations and env steps it used, its wall time, the first evaluation that completed a lap
(iteration, env steps, seconds of wall time), the fastest evaluation lap seen at any evaluation,
and the final evaluation (laps, best lap, return). Evaluations are the deterministic episodes
in racer/evaluate.py, logged every ``eval_every`` iterations and on the last one; rows between
evaluations have empty eval columns and are skipped here.

Run: uv run python docs/lessons/scripts/run_summary.py runs/evolution-*
"""

import json
import math
import sys
from pathlib import Path

from racer.runlog import RunDir


def summarize(path: Path) -> str:
    config = json.loads((path / "config.json").read_text())
    train, env = config["train"], config["env"]
    rows = RunDir.open(path).read_metrics()
    evals = [r for r in rows if not math.isnan(r["eval_laps"])]
    last = rows[-1]
    first = next((r for r in evals if r["eval_laps"] >= 1), None)
    laps = [r["eval_best_lap_s"] for r in evals if not math.isnan(r["eval_best_lap_s"])]
    settings = (f"pop {train.get('population', '-')}, elite {train.get('elite', '-')}, "
                f"std {train.get('mutation_std', '-')}, rays {env['n_rays']}")
    lines = [f"{path.name}  ({config['method']} on {train['track']}, {settings}, "
             f"seed {config['seed']})",
             f"  {int(last['iteration'])} iterations, {int(last['env_steps']):,} env steps, "
             f"{last['wall_time_s']:.0f} s"]
    if first is None:
        lines.append("  first lap: never")
    else:
        lines.append(f"  first lap: iteration {int(first['iteration'])}, "
                     f"{int(first['env_steps']):,} steps, {first['wall_time_s']:.0f} s")
    if laps:
        best = min(range(len(laps)), key=laps.__getitem__)
        it = int([r for r in evals if not math.isnan(r["eval_best_lap_s"])][best]["iteration"])
        lines.append(f"  fastest eval lap: {laps[best]:.2f} s (iteration {it})")
    final = evals[-1]
    lines.append(f"  final eval (iteration {int(final['iteration'])}): "
                 f"{int(final['eval_laps'])} laps, best lap {final['eval_best_lap_s']:.2f} s, "
                 f"progress {final['eval_progress_m']:.1f} m, return {final['eval_return']:.2f}")
    return "\n".join(lines)


def main() -> None:
    for arg in sys.argv[1:]:
        print(summarize(Path(arg)))


if __name__ == "__main__":
    main()
