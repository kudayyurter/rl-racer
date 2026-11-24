"""Lesson 08: put one run's fastest lap on screen as a ghost next to another run's live car.

``racer watch`` replays the ghosts of the run it is watching. To race two runs against each
other this script builds a scratch run directory (``runs/ghostcmp-<name>``, git-ignored like
every run) that the viewer can open unchanged:

  - config.json, metrics.csv and the live car's checkpoint come from LIVE (``RUN@ITERATION``
    picks a checkpoint; default the final one), so the status block and curves are LIVE's;
  - ghosts/ holds the fastest recorded lap of each GHOST run, with its original iteration.

Then: uv run racer watch runs/ghostcmp-<name> --ghosts 2

Run: uv run python docs/lessons/scripts/ghost_pair.py NAME LIVE[@ITERATION] GHOST_RUN [GHOST_RUN ...]
"""

import shutil
import sys
from pathlib import Path

import numpy as np


def fastest_ghost(run: Path) -> Path:
    best, best_t = None, float("inf")
    for p in (run / "ghosts").glob("ghost_*.npz"):
        with np.load(p) as z:
            if float(z["lap_time"]) < best_t:
                best, best_t = p, float(z["lap_time"])
    if best is None:
        raise SystemExit(f"{run} has no ghost laps")
    return best


def main() -> None:
    name, live, *ghost_runs = sys.argv[1:]
    run, _, iteration = live.partition("@")
    run = Path(run)
    out = Path("runs") / f"ghostcmp-{name}"
    if out.exists():
        shutil.rmtree(out)
    (out / "checkpoints").mkdir(parents=True)
    (out / "ghosts").mkdir()
    for f in ("config.json", "metrics.csv"):
        shutil.copy(run / f, out / f)
    ckpt = f"ckpt_{int(iteration):05d}.pt" if iteration else "latest.pt"
    shutil.copy(run / "checkpoints" / ckpt, out / "checkpoints" / "latest.pt")
    print(f"live car: {run / 'checkpoints' / ckpt}")
    for g in ghost_runs:
        src = fastest_ghost(Path(g))
        dst = out / "ghosts" / src.name
        if dst.exists():   # two runs' best laps at the same iteration: keep both
            dst = out / "ghosts" / f"{src.stem}b.npz"
        shutil.copy(src, dst)
        with np.load(src) as z:
            print(f"ghost: {src}  ({float(z['lap_time']):.2f} s, iteration {int(z['iteration'])})")
    print(f"wrote {out}; watch it with: uv run racer watch {out} --ghosts {len(ghost_runs)}")


if __name__ == "__main__":
    main()
