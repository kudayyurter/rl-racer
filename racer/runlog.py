"""Run directories: where every training run writes its config, metrics, checkpoints and ghosts.

Why this exists: all learners share one on-disk layout, so the ``watch`` viewer, the plotting
scripts and the lessons can read any run without knowing which method produced it.

    runs/<method>-<YYYYmmdd-HHMMSS>-s<seed>/
        config.json            what the run was started with
        metrics.csv            one row per logging step (METRIC_COLUMNS, plus method extras)
        checkpoints/           ckpt_XXXXX.pt per save, and latest.pt (a copy of the newest)
        ghosts/                ghost_XXXXX.npz: the best evaluation lap at that iteration

Training and the viewer run at the same time, so every file the viewer reads is written
atomically: the bytes go to a hidden temp file in the same directory, then ``os.replace``
swaps it into place. A reader therefore sees either the old file or the new one, never half
of one. The temp names start with a dot and end in ``.tmp`` so they match none of the
patterns the viewer looks for.
"""

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

METRIC_COLUMNS = ["iteration", "env_steps", "wall_time_s", "train_return_mean", "eval_return",
                  "eval_progress_m", "eval_laps", "eval_best_lap_s"]


@dataclass
class Ghost:
    """The best evaluation lap at one iteration, for the viewer to replay as a ghost car."""

    iteration: int
    t: np.ndarray        # (T,) seconds since the lap started
    poses: np.ndarray    # (T, 3) x, y, heading
    lap_time: float


def _atomic_write(path: Path, write) -> None:
    """Call ``write(tmp_path)`` to produce a file, then move it onto ``path`` in one step.

    The temp file lives in the same directory because ``os.replace`` is only atomic within
    one filesystem. If ``write`` fails, the temp file is removed and the error re-raised,
    leaving any previous ``path`` untouched.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class RunDir:
    """One run's directory. Create a new one with ``create``, reopen an old one with ``open``."""

    def __init__(self, path: Path, columns: list[str]):
        self.path = Path(path)
        self.columns = columns

    @classmethod
    def create(cls, method: str, seed: int, root: str | Path = "runs",
               extra_columns: tuple[str, ...] = ()) -> "RunDir":
        """Make ``<root>/<method>-<timestamp>-s<seed>``; the CSV gets ``extra_columns`` too."""
        from datetime import datetime  # local: only create() needs the clock

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(root) / f"{method}-{stamp}-s{seed}"
        path, n = base, 1
        # Two runs started in the same second must not share (and overwrite) a directory.
        while True:
            try:
                path.mkdir(parents=True)
                break
            except FileExistsError:
                n += 1
                path = base.with_name(f"{base.name}-{n}")
        (path / "checkpoints").mkdir()
        (path / "ghosts").mkdir()
        return cls(path, METRIC_COLUMNS + list(extra_columns))

    @classmethod
    def open(cls, path: str | Path) -> "RunDir":
        """Reopen an existing run, recovering the metric columns from its CSV header."""
        path = Path(path)
        csv_path = path / "metrics.csv"
        columns = list(METRIC_COLUMNS)
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                header = next(csv.reader(f), None)
            if header:
                columns = header
        # Read-only: watch and compare open runs, so nothing is created here.
        return cls(path, columns)

    # ---------------------------------------------------------------- config and metrics

    def save_config(self, d: dict) -> None:
        """Write the run's settings to ``config.json``."""
        (self.path / "config.json").write_text(json.dumps(d, indent=2, default=str))

    def log(self, row: dict) -> None:
        """Append one row to ``metrics.csv`` (header written on first use).

        Every METRIC_COLUMNS key is required. Keys that are not declared columns raise
        instead of being dropped, so a typo cannot silently lose a metric. Declared extra
        columns may be omitted; they are written empty.
        """
        required = set(METRIC_COLUMNS)
        missing = sorted(required - set(row))
        unknown = sorted(set(row) - set(self.columns))
        if missing:
            raise ValueError(f"metrics row is missing column(s): {', '.join(missing)}")
        if unknown:
            raise ValueError(f"metrics row has unknown column(s): {', '.join(unknown)}")
        csv_path = self.path / "metrics.csv"
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def read_metrics(self) -> list[dict]:
        """All logged rows as dicts of floats (an empty cell reads as NaN).

        The viewer reads this file while the trainer appends to it, so the last row can be
        caught half written. ``DictReader`` fills the cells such a row is missing with None
        (and files surplus cells under the key None); those rows are skipped; the next read
        will see them whole.
        """
        csv_path = self.path / "metrics.csv"
        if not csv_path.exists():
            return []
        with open(csv_path, newline="") as f:
            return [{k: float(v) if v != "" else float("nan") for k, v in r.items()}
                    for r in csv.DictReader(f)
                    if None not in r and None not in r.values()]

    # ---------------------------------------------------------------- checkpoints and ghosts

    def save_checkpoint(self, ckpt: dict, iteration: int) -> None:
        """Save ``checkpoints/ckpt_XXXXX.pt`` and refresh ``checkpoints/latest.pt``, atomically."""
        folder = self.path / "checkpoints"
        _atomic_write(folder / f"ckpt_{iteration:05d}.pt", lambda p: torch.save(ckpt, p))
        _atomic_write(folder / "latest.pt", lambda p: torch.save(ckpt, p))

    def save_ghost(self, iteration: int, t: np.ndarray, poses: np.ndarray,
                   lap_time: float) -> None:
        """Save a best-lap recording to ``ghosts/ghost_XXXXX.npz``, atomically."""
        def write(tmp: str) -> None:
            # A file object, because np.savez would append ".npz" to a bare temp name.
            with open(tmp, "wb") as f:
                np.savez(f, t=np.asarray(t), poses=np.asarray(poses),
                         lap_time=float(lap_time), iteration=iteration)
        _atomic_write(self.path / "ghosts" / f"ghost_{iteration:05d}.npz", write)

    def ghosts(self) -> list[Ghost]:
        """All saved ghosts, oldest iteration first."""
        out = []
        for p in sorted((self.path / "ghosts").glob("ghost_*.npz")):
            with np.load(p) as z:
                out.append(Ghost(int(z["iteration"]), z["t"], z["poses"], float(z["lap_time"])))
        return sorted(out, key=lambda g: g.iteration)


def latest_checkpoint(run_path: str | Path) -> Path | None:
    """Path of the newest checkpoint in a run, or None if it has not saved one yet."""
    p = Path(run_path) / "checkpoints" / "latest.pt"
    return p if p.exists() else None
