"""Compare finished runs: learning curves, success rates and best laps across methods.

Why this exists: the lessons quote numbers ("PPO laps in N steps, evolution in M"), and those
numbers must come from one place that groups runs correctly. ``racer compare`` reads run
directories, groups them, and writes two plots plus a markdown summary table.

Grouping. Runs are only comparable if they trained the same way on the same problem, so a
group is ``(method, track, env-config hash, train-config hash)``: seeds of one method on one
track with an identical environment config and identical method settings. The train hash leaves
out the seed, the one setting that is meant to differ between the runs of a group. Change a
reward weight, a mutation std or the budget and the runs land in a separate group.

Eval points. ``metrics.csv`` has a row per logging step, but only some rows carry an
evaluation (the others have NaN eval columns). Only rows with a finite ``eval_progress_m``
are eval points; every curve here is drawn at eval points only.

Aggregation across seeds. Each seed's curve is a step function (an eval result stays true
until the next eval), so on a common x grid every seed holds its last value. Beyond a seed's
final point it contributes nothing rather than being extrapolated. The plot shows the median
with a min-max band. For "best lap so far", seeds that have not lapped yet are left out of
the median at that x, so early on the line describes only the lucky seeds; the success rate
in the summary table is what says how many seeds that is.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: we only write PNG files
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402

from racer.runlog import RunDir  # noqa: E402

# One colour per method, shared by both plots so a method is recognisable at a glance.
METHOD_COLORS = {"evolution": "#d95f02", "reinforce": "#7570b3", "a2c": "#1b9e77",
                 "ppo": "#e7298a", "sb3": "#1f78b4"}
_FALLBACK_COLORS = ["#66a61e", "#a6761d", "#666666"]
_LINESTYLES = ["-", "--", ":", "-."]

X_AXES = {"steps": "Environment steps", "time": "Wall-clock time (s)"}


def env_hash(env_config: dict) -> str:
    """Short stable fingerprint of an env config (key order does not matter)."""
    return hashlib.sha1(json.dumps(env_config, sort_keys=True).encode()).hexdigest()[:6]


# Train-config keys that differ between the seeds of one experiment and so must not split a group.
PER_RUN_TRAIN_KEYS = ("seed",)


def train_hash(train_config: dict) -> str:
    """Fingerprint of a method's settings, ignoring the per-run keys (the seed)."""
    shared = {k: v for k, v in train_config.items() if k not in PER_RUN_TRAIN_KEYS}
    return env_hash(shared)   # same recipe: sorted JSON, short sha1


def load_run(path: str | Path) -> dict:
    """One run's eval-point curves plus the identifying fields used for grouping.

    ``best_lap`` is the running minimum of ``eval_best_lap_s``: NaN until the first lap, and
    it never gets worse. Returned arrays all have one entry per eval point.
    """
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    rows = [r for r in RunDir.open(path).read_metrics() if np.isfinite(r["eval_progress_m"])]

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=float)

    best = col("eval_best_lap_s")
    # fmin ignores NaN, so a NaN row keeps the previous best; leading NaNs stay NaN.
    running = np.fmin.accumulate(best) if len(best) else best
    return {
        "path": path, "method": config["method"], "seed": config["seed"],
        "track": config["train"]["track"], "env_hash": env_hash(config["env"]),
        "train_hash": train_hash(config["train"]),
        "steps": col("env_steps"), "time": col("wall_time_s"),
        "progress": col("eval_progress_m"), "laps": col("eval_laps"), "best_lap": running,
    }


def group_runs(curves: list[dict]) -> dict[tuple, list[dict]]:
    """Group loaded runs by (method, track, env hash, train hash), in a stable order."""
    def key(c):
        return (c["method"], c["track"], c["env_hash"], c["train_hash"])

    groups: dict[tuple, list[dict]] = {}
    for c in sorted(curves, key=lambda c: (*key(c), c["seed"])):
        groups.setdefault(key(c), []).append(c)
    return groups


def group_labels(keys) -> dict[tuple, str]:
    """Readable names: "ppo · twisty", plus hashes only where method and track collide.

    Among the groups that share a method and track, "#<env hash>" is added if their env configs
    differ and "train#<train hash>" if their method settings differ, so each label names what
    actually tells that group apart.
    """
    keys = list(keys)
    labels = {}
    for method, track, eh, th in keys:
        clash = [k for k in keys if k[:2] == (method, track)]
        label = f"{method} · {track}"
        if len({k[2] for k in clash}) > 1:
            label += f" #{eh}"
        if len({k[3] for k in clash}) > 1:
            label += f" train#{th}"
        labels[(method, track, eh, th)] = label
    return labels


def aggregate(curves: list[dict], x_key: str, y_key: str):
    """Median, min and max across seeds of ``y_key`` on the union of the seeds' x values.

    Returns ``(x, median, lo, hi, n)`` where ``n`` counts the seeds contributing at each x. Each seed holds its last value until its next point and
    is absent outside its own x range; NaN y values (no lap yet) are left out too. Where no
    seed contributes the result is NaN.
    """
    x = np.unique(np.concatenate([c[x_key] for c in curves if len(c[x_key])]))
    stack = np.full((len(curves), len(x)), np.nan)
    for i, c in enumerate(curves):
        if not len(c[x_key]):
            continue
        idx = np.searchsorted(c[x_key], x, side="right") - 1   # last point at or before x
        inside = (idx >= 0) & (x <= c[x_key][-1])
        stack[i, inside] = c[y_key][idx[inside]]
    median, lo, hi = (np.full(len(x), np.nan) for _ in range(3))
    n = np.zeros(len(x), dtype=int)
    for j in range(len(x)):
        vals = stack[:, j][np.isfinite(stack[:, j])]
        n[j] = len(vals)
        if len(vals):
            median[j], lo[j], hi[j] = np.median(vals), vals.min(), vals.max()
    return x, median, lo, hi, n


def _first_lap(curve: dict):
    """(env_steps, wall_time_s) at the first eval with a lap, or None if the seed never lapped."""
    hit = np.flatnonzero(curve["laps"] >= 1)
    return (curve["steps"][hit[0]], curve["time"][hit[0]]) if len(hit) else None


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def summarize(runs: list) -> list[dict]:
    """One summary dict per group of run directories (the pure, testable core).

    A seed succeeds if any eval row has at least one lap. Best-lap and time-to-first-lap
    statistics use successful seeds only; they are NaN when no seed succeeded.
    """
    groups = group_runs([load_run(p) for p in runs])
    labels = group_labels(groups)
    out = []
    for key, curves in groups.items():
        ok = [c for c in curves if _first_lap(c) is not None]
        bests = [float(c["best_lap"][-1]) for c in ok]
        firsts = [_first_lap(c) for c in ok]
        out.append({
            "label": labels[key], "method": key[0], "track": key[1], "env_hash": key[2],
            "train_hash": key[3],
            "seeds": len(curves), "success": f"{len(ok)}/{len(curves)}",
            "best_lap_median": _median(bests),
            "best_lap_min": min(bests) if bests else float("nan"),
            "best_lap_max": max(bests) if bests else float("nan"),
            "steps_to_lap": _median([f[0] for f in firsts]),
            "time_to_lap": _median([f[1] for f in firsts]),
            "curves": curves,
        })
    return out


# ------------------------------------------------------------------ output

def _fmt(v: float, spec: str) -> str:
    return "–" if not np.isfinite(v) else format(v, spec)


def summary_markdown(rows: list[dict], name: str, today: date | None = None) -> str:
    """The markdown block appended to results/summary.md: a dated heading and one table."""
    lines = [f"## {name} ({(today or date.today()).isoformat()})", "",
             "| group | seeds | success | best lap s, median (range) "
             "| env steps to first lap | wall s to first lap |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        best = "–" if not np.isfinite(r["best_lap_median"]) else (
            f"{r['best_lap_median']:.2f} ({r['best_lap_min']:.2f}–{r['best_lap_max']:.2f})")
        lines.append(f"| {r['label']} | {r['seeds']} | {r['success']} | {best} "
                     f"| {_fmt(r['steps_to_lap'], ',.0f')} | {_fmt(r['time_to_lap'], '.1f')} |")
    return "\n".join(lines) + "\n\n"


def _styles(rows: list[dict]) -> dict[str, tuple[str, str]]:
    """label -> (colour, linestyle): colour by method, linestyle to tell a method's groups apart."""
    styles, seen, extra = {}, {}, {}
    for r in rows:
        color = METHOD_COLORS.get(r["method"])
        if color is None:
            color = extra.setdefault(r["method"], _FALLBACK_COLORS[len(extra) % len(_FALLBACK_COLORS)])
        n = seen[r["method"]] = seen.get(r["method"], -1) + 1
        styles[r["label"]] = (color, _LINESTYLES[n % len(_LINESTYLES)])
    return styles


def plot_comparison(rows: list[dict], x_name: str, path: Path) -> None:
    """Two panels (best lap so far, eval progress) against ``x_name`` ('steps' or 'time')."""
    x_label = X_AXES[x_name]
    styles = _styles(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    panels = [("best_lap", "Best lap so far (s, lower is better)\n"
                           "median over seeds that have lapped; faint = not all seeds yet"),
              ("progress", "Eval progress (m along the track)")]
    for ax, (y_key, y_label) in zip(axes, panels):
        for r in rows:
            x, med, lo, hi, n = aggregate(r["curves"], x_name, y_key)
            if not np.isfinite(med).any():
                continue
            color, ls = styles[r["label"]]
            # Seeds still running at each x: where fewer than that have a value (best lap
            # before every seed has lapped) the line is drawn faint and dashed.
            active = aggregate(r["curves"], x_name, "progress")[4]
            full = n >= active
            label = f"{r['label']} (n={len(r['curves'])})"
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, step="post")
            ax.step(x, med, where="post", color=color, linestyle="--", linewidth=1.5,
                    alpha=0.45, label=None if full.any() else label)
            if full.any():
                ax.step(x, np.where(full, med, np.nan), where="post", color=color,
                        linestyle=ls, linewidth=2, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
        if x_name == "steps":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    # A group with no lap yet only shows in the progress panel, so build one legend from both.
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    axes[0].legend(handles.values(), handles.keys(), fontsize=9)
    fig.suptitle("Median across seeds, band = min to max", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_compare(run_dirs: list[str], out: str | Path, name: str) -> int:
    """Write the two plots and append the summary table. Returns a process exit code."""
    rows = summarize(run_dirs)
    if not rows:
        print("racer compare: no run directories given")
        return 1
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for x_name in X_AXES:
        plot_comparison(rows, x_name, out / f"{name}_vs_{x_name}.png")
    with open(out / "summary.md", "a") as f:
        f.write(summary_markdown(rows, name))
    print(f"wrote {out}/{name}_vs_steps.png, {out}/{name}_vs_time.png and appended {out}/summary.md")
    return 0
