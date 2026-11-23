"""Simulator throughput benchmark: how many car-decisions per second does BatchSim manage?

Why this exists: training time is dominated by simulation, and the whole point of batching is
that stepping 256 cars costs far less than 256 times one car. This measures that directly, so
it is easy to see what batch size to train with and to catch performance regressions.

A "car-step" is one decision (``action_repeat`` physics substeps plus sensing) for one car.
Cars drive random actions on the oval with auto-reset, so crashes and respawns are included,
as they are during training.
"""

import time

import numpy as np

from racer.config import EnvConfig
from racer.sim import BatchSim
from racer.track import load_track


def bench(n_cars_list=(1, 16, 64, 128, 256), seconds: float = 3.0) -> list[dict]:
    """Run each batch size for about ``seconds`` of wall time; return one dict per size."""
    track = load_track("oval")
    cfg = EnvConfig()
    rng = np.random.default_rng(0)
    results = []
    for n in n_cars_list:
        sim = BatchSim(track, cfg, n, seed=0, auto_reset=True)
        sim.reset()
        sim.step(rng.uniform(-1.0, 1.0, (n, 2)))  # warm-up, so first-call costs don't count
        decisions = 0
        start = time.perf_counter()
        elapsed = 0.0
        while elapsed < seconds:
            sim.step(rng.uniform(-1.0, 1.0, (n, 2)))
            decisions += 1
            elapsed = time.perf_counter() - start
        results.append({
            "n_cars": n,
            "decisions": decisions,
            "seconds": elapsed,
            "car_steps_per_s": decisions * n / elapsed,
        })
    return results


def format_table(results: list[dict]) -> str:
    """Render bench() results as a plain-text table."""
    lines = [f"{'cars':>6} {'decisions':>10} {'seconds':>8} {'car-steps/s':>12}"]
    for r in results:
        lines.append(f"{r['n_cars']:>6} {r['decisions']:>10} {r['seconds']:>8.2f} "
                     f"{r['car_steps_per_s']:>12,.0f}")
    return "\n".join(lines)
