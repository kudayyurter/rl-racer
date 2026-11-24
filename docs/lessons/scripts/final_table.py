"""Lesson 10: the per-run table and the "who is ahead at step X / second T" table.

Per run (three lines each): env steps, wall time and steps per second; how much of the wall time
went into evaluating and checkpointing, and the steps per second of the rest; how many other training runs were running at the same
time, on average; the first evaluation that lapped (env steps and seconds); the fastest evaluation
lap; how many evaluations lapped; and the final evaluation.

Evaluation time. Each trainer writes ``wall_time_s`` into a row *before* it runs that row's
evaluation and saves the checkpoint, so that work shows up as extra time on the *next* row. The
script takes the median time of a row that does not follow an evaluation as the training cost
of one iteration, and adds up how much longer the rows after evaluations took. SB3 logs a row only
at evaluations, so there is no baseline row and its figure is "n/a".

That last number matters for every wall-clock comparison. The run directories record when each
run started (the timestamp in the name) and how long it took (``wall_time_s`` in the last row of
``metrics.csv``), so overlap can be measured after the fact. It counts only runs whose directories
are still under ``runs/``; any other load on the machine (a test suite, a process started by
mistake) is not in it, so it is a lower bound on how crowded the machine was.

With ``--at``, it prints instead, for each group of runs (method, track), the median over seeds
of "best lap so far" at a few env-step budgets and a few wall-clock times, plus how many seeds had
lapped by then. That is the ranking the two comparison plots show, as numbers, with one
difference: a run that has already finished keeps its final best lap (marked ``*``), because
after T seconds you would have that run's result in hand. (``racer compare``'s plots instead end
each seed's line where the run ended.)

Run: uv run python docs/lessons/scripts/final_table.py RUN_DIR ...
     uv run python docs/lessons/scripts/final_table.py --at RUN_DIR ...
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from racer.compare import group_runs, load_run
from racer.runlog import RunDir

STEP_POINTS = [250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_500_000]
TIME_POINTS = [30, 60, 120, 300, 450, 900]


def _span(path: Path) -> tuple[float, float] | None:
    """(start, end) in POSIX seconds, from the directory name and the final wall time."""
    parts = path.name.split("-")  # method-YYYYMMDD-HHMMSS-sN
    try:
        start = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S").timestamp()
    except (IndexError, ValueError):
        return None  # a hand-named directory (e.g. a copy): no start time to read
    rows = RunDir.open(path).read_metrics()
    return start, start + rows[-1]["wall_time_s"]


def mean_overlap(path: Path, all_runs: list[Path]) -> float:
    """Time-averaged number of OTHER runs that were training while this one was."""
    me = _span(path)
    if me is None:
        return float("nan")
    total = 0.0
    for other in all_runs:
        if other.resolve() == path.resolve():
            continue
        span = _span(other)
        if span is None:
            continue
        total += max(0.0, min(me[1], span[1]) - max(me[0], span[0]))
    return total / (me[1] - me[0])


def eval_seconds(rows: list[dict]) -> float:
    """Seconds spent evaluating and checkpointing, read from the row timings (see docstring)."""
    wall = np.array([r["wall_time_s"] for r in rows])
    evaluated = np.array([not math.isnan(r["eval_laps"]) for r in rows])
    dt = np.diff(wall)            # dt[i]: time between row i and row i + 1
    after_eval = evaluated[:-1]   # row i evaluated, so dt[i] carries that evaluation
    if after_eval.all() or not after_eval.any():
        return float("nan")       # no plain row to compare with (SB3), or no evaluation at all
    base = np.median(dt[~after_eval])
    return float((dt[after_eval] - base).sum())


def run_line(path: Path, all_runs: list[Path]) -> str:
    config = json.loads((path / "config.json").read_text())
    rows = RunDir.open(path).read_metrics()
    evals = [r for r in rows if not math.isnan(r["eval_laps"])]
    last, final = rows[-1], evals[-1]
    first = next((r for r in evals if r["eval_laps"] >= 1), None)
    laps = [r["eval_best_lap_s"] for r in evals if not math.isnan(r["eval_best_lap_s"])]
    lapped = sum(1 for r in evals if r["eval_laps"] >= 1)
    first_txt = (f"{int(first['env_steps']):,} steps / {first['wall_time_s']:.0f} s"
                 if first else "never")
    final_txt = (f"{int(final['eval_laps'])} laps, best {final['eval_best_lap_s']:.2f} s"
                 if final["eval_laps"] >= 1 else f"0 laps, {final['eval_progress_m']:.0f} m")
    fastest_txt = f"{min(laps):.2f} s" if laps else "-"
    ev = eval_seconds(rows)
    if math.isnan(ev):
        ev_txt = "n/a"
    elif ev <= 0:
        # Evolution's generations vary in length far more than its few evaluations cost.
        ev_txt = "about 0 s (less than the row-to-row noise)"
    else:
        ev_txt = (f"{ev:.0f} s ({ev / last['wall_time_s']:.0%}), so training alone ran at "
                  f"{last['env_steps'] / (last['wall_time_s'] - ev) / 1000:.1f}k steps/s")
    return (f"{path.name}  {config['method']} on {config['train']['track']}, seed {config['seed']}\n"
            f"  {int(last['env_steps']):,} steps in {last['wall_time_s']:.0f} s = "
            f"{last['env_steps'] / last['wall_time_s'] / 1000:.1f}k steps/s; "
            f"evaluating {len(evals)} times took {ev_txt}; other runs at the same time {mean_overlap(path, all_runs):.1f} on average\n"
            f"  first lap: {first_txt}; fastest lap: {fastest_txt}; "
            f"evaluations that lapped: {lapped} of {len(evals)}; final: {final_txt}")


def best_at(curve: dict, x_key: str, x: float) -> float:
    """A seed's best lap so far at x (NaN if it has not lapped by then). Holds its last value."""
    i = np.searchsorted(curve[x_key], x, side="right") - 1
    return float(curve["best_lap"][i]) if i >= 0 else float("nan")


def at_table(paths: list[Path]) -> str:
    groups = group_runs([load_run(p) for p in paths])
    out = []
    for x_key, points, unit in (("steps", STEP_POINTS, "env steps"), ("time", TIME_POINTS, "s")):
        out.append(f"best lap so far, median over seeds that had lapped (lapped/seeds), at {unit}:")
        out.append(f"{'group':<22}" + "".join(f"{p:>16,}" for p in points))
        for (method, track, *_), curves in groups.items():
            cells = []
            for p in points:
                vals = [v for v in (best_at(c, x_key, p) for c in curves) if np.isfinite(v)]
                # 2% slack on steps: budgets overshoot a little, so 4.5M counts as reached.
                ended = any(p > c[x_key][-1] * (1.02 if x_key == "steps" else 1.0) for c in curves)
                med = f"{np.median(vals):.2f}" if vals else "-"
                cells.append(f"{med} ({len(vals)}/{len(curves)})" + ("*" if ended else ""))
            out.append(f"{method + ' · ' + track:<22}" + "".join(f"{c:>16}" for c in cells))
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    at = "--at" in args
    paths = [Path(a) for a in args if a != "--at"]
    if at:
        print(at_table(paths))
    else:
        all_runs = sorted(p for p in Path("runs").iterdir() if (p / "metrics.csv").exists())
        for p in paths:
            print(run_line(p, all_runs))
