"""Lessons 05-07: every evaluation of a run on one screen, and how steady it was.

For each run directory: one token per evaluation row of metrics.csv, ``iteration:laps``
followed by the best lap in seconds when it lapped, or by the progress in meters when it did not
(``25:2(32.3s)``, ``40:0[582m]``). Then a summary: how many evaluations completed at least one
lap, how many evaluations after the first lapping one did NOT (a policy that had learned to lap
and then lost it, at least for that one deterministic episode), the final evaluation, and the
explained variance (A2C, PPO) and PPO's approx_kl and clip_frac as quartiles over all iterations
and over the last tenth of them.

Run: uv run python docs/lessons/scripts/eval_history.py runs/a2c-<timestamp>-s0 [--short]
     (--short: summary only, no per-evaluation tokens)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from racer.runlog import RunDir


def summarize(path: Path, short: bool) -> str:
    config = json.loads((path / "config.json").read_text())
    rows = RunDir.open(path).read_metrics()
    evals = [r for r in rows if not math.isnan(r["eval_laps"])]
    tokens = []
    for r in evals:
        tail = (f"({r['eval_best_lap_s']:.1f}s)" if r["eval_laps"] > 0
                else f"[{r['eval_progress_m']:.0f}m]")
        tokens.append(f"{int(r['iteration'])}:{int(r['eval_laps'])}{tail}")
    lapped = [r["eval_laps"] > 0 for r in evals]
    first = lapped.index(True) if any(lapped) else None
    lost = sum(1 for x in lapped[first:] if not x) if first is not None else 0
    after = len(lapped) - first if first is not None else 0
    final = evals[-1]
    out = [f"{path.name}  ({config['method']} on {config['train']['track']}, seed {config['seed']}, "
           f"{int(rows[-1]['iteration'])} iterations, {int(rows[-1]['env_steps']):,} steps, "
           f"{rows[-1]['wall_time_s']:.0f} s)"]
    if not short:
        out.append("  " + " ".join(tokens))
    first_txt = (f"first at iteration {int(evals[first]['iteration'])} "
                 f"({int(evals[first]['env_steps']):,} steps)" if first is not None else "never")
    out.append(f"  lapping evaluations: {sum(lapped)} of {len(evals)}; {first_txt}; "
               f"evaluations from then on that did not lap: {lost} of {after}")
    out.append(f"  final evaluation: {int(final['eval_laps'])} laps, "
               f"progress {final['eval_progress_m']:.1f} m, return {final['eval_return']:.2f}"
               + (f", best lap {final['eval_best_lap_s']:.2f} s" if final["eval_laps"] > 0 else ""))
    q = lambda x, f="+.2f": "/".join(f"{v:{f}}" for v in np.percentile(x, [25, 50, 75]))
    for column, name, fmt in (("explained_variance", "explained variance", "+.2f"),
                              ("approx_kl", "approx_kl", ".4f"), ("clip_frac", "clip_frac", ".3f")):
        if column not in rows[0]:
            continue
        vals = np.array([r[column] for r in rows])
        vals = vals[~np.isnan(vals)]
        tail = vals[-max(1, len(vals) // 10):]
        out.append(f"  {name} quartiles: all iterations {q(vals, fmt)}, last tenth {q(tail, fmt)}")
    return "\n".join(out)


def main() -> None:
    short = "--short" in sys.argv
    for arg in [a for a in sys.argv[1:] if a != "--short"]:
        print(summarize(Path(arg), short))


if __name__ == "__main__":
    main()
