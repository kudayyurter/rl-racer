"""A small live line chart of one metrics.csv column against training env steps.

Why this exists: the watch viewer's car shows how the *current* policy drives; the curve shows
how it got there. Plotting against env steps (car decisions simulated) rather than iterations
keeps the x axis comparable between methods, whose iterations mean very different amounts of
experience.

Rows whose value is NaN are skipped: evaluation columns are only filled on evaluation
iterations, and a lap time stays NaN until the policy completes a lap.
"""

import math

import numpy as np
import pyray as rl

AXIS = (70, 76, 88, 255)
LINE = (250, 192, 28, 255)        # the live car's yellow: this is its history
TITLE = (220, 224, 232, 255)
MUTED = (150, 156, 168, 255)
TITLE_SIZE = 20
TICK_SIZE = 10
PLOT_TOP = 30                     # pixels under the rect's top edge reserved for the title
PLOT_BOTTOM = 16                  # pixels above the bottom edge reserved for the x labels

# Pretty names and units for the columns the viewer plots; others show their raw name.
NAMES = {"eval_best_lap_s": ("best eval lap", "s"),
         "eval_progress_m": ("eval progress", "m"),
         "eval_return": ("eval return", ""),
         "train_return_mean": ("mean train return", "")}


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of the rows that have a value for ``key``; x is env steps (or the row index)."""
    xs, ys = [], []
    for n, row in enumerate(rows):
        value = row.get(key, math.nan)
        if not math.isnan(value):
            xs.append(row.get("env_steps", float(n)))
            ys.append(value)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def _span(lo: float, hi: float) -> tuple[float, float]:
    """Widen a value range a little, and give a single value a non-zero range."""
    if hi - lo < 1e-9:
        pad = max(abs(hi) * 0.05, 1.0)
    else:
        pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def curve_points(rows: list[dict], key: str, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Screen points (K, 2) of the ``key`` series fitted inside ``rect`` (x, y, w, h).

    x runs from 0 env steps at the left edge to the latest row's env steps at the right edge;
    y runs from a padded minimum at the bottom to a padded maximum at the top.
    """
    xs, ys = _series(rows, key)
    if len(xs) == 0:
        return np.zeros((0, 2))
    x, y, w, h = rect
    x_max = max(xs.max(), 1.0)
    lo, hi = _span(ys.min(), ys.max())
    sx = x + w * xs / x_max
    sy = y + h * (1.0 - (ys - lo) / (hi - lo))   # screen y grows downwards
    return np.stack([sx, sy], axis=1)


def _fmt(value: float) -> str:
    """Short number for axis labels: 28.4, 1428, 507k, 1.2M."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e4:
        return f"{value / 1e3:.0f}k"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def draw_curve(rows: list[dict], key: str, rect: tuple[float, float, float, float]) -> None:
    """Draw the ``key`` column of ``rows`` as a titled line chart filling ``rect``."""
    x, y, w, h = rect
    name, unit = NAMES.get(key, (key, ""))
    xs, ys = _series(rows, key)
    title = name if len(ys) == 0 else f"{name}  {_fmt(ys[-1])} {unit}".rstrip()
    rl.draw_text(title, int(x), int(y), TITLE_SIZE, TITLE)

    plot = (x + 40, y + PLOT_TOP, w - 40, h - PLOT_TOP - PLOT_BOTTOM)
    px, py, pw, ph = plot
    rl.draw_line_ex((px, py + ph), (px + pw, py + ph), 1.0, AXIS)
    rl.draw_line_ex((px, py), (px, py + ph), 1.0, AXIS)
    if len(ys) == 0:
        rl.draw_text("no evaluation yet", int(px + 12), int(py + ph / 2 - 10), TITLE_SIZE, MUTED)
        return

    lo, hi = _span(ys.min(), ys.max())
    rl.draw_text(_fmt(hi), int(x), int(py), TICK_SIZE, MUTED)
    rl.draw_text(_fmt(lo), int(x), int(py + ph - TICK_SIZE), TICK_SIZE, MUTED)
    rl.draw_text("0", int(px), int(py + ph + 4), TICK_SIZE, MUTED)
    last = f"{_fmt(max(xs.max(), 1.0))} env steps"
    rl.draw_text(last, int(px + pw - rl.measure_text(last, TICK_SIZE)), int(py + ph + 4),
                 TICK_SIZE, MUTED)

    pts = curve_points(rows, key, plot)
    for a, b in zip(pts[:-1].tolist(), pts[1:].tolist()):
        rl.draw_line_ex(tuple(a), tuple(b), 2.0, LINE)
    for p in pts.tolist():
        rl.draw_circle_v(tuple(p), 3.0, LINE)
