"""Lesson 03 experiment: how does action_repeat change simulator throughput?

``racer bench`` always uses the default EnvConfig (action_repeat = 4). This script runs the same
measurement loop as ``racer.bench.bench`` (random actions on the oval, auto-reset on) with
action_repeat set to 2, 4 and 8, at two batch sizes.

Two rates are printed, because they answer different questions:

* car-steps/s       agent decisions per second, the unit every step budget is counted in
* sim-s per wall-s  simulated driving time per second of wall time (car-steps/s * action_repeat
                    * dt): how fast the cars cover the track, whatever the decision rate

Run: uv run python docs/lessons/scripts/bench_action_repeat.py   (about 20 s)
"""

import dataclasses
import time

import numpy as np

from racer.config import EnvConfig
from racer.sim import BatchSim
from racer.track import load_track

REPEATS = (2, 4, 8)
N_CARS = (16, 256)
SECONDS = 3.0


def measure(cfg: EnvConfig, n: int) -> float:
    """car-steps per second, measured exactly like racer.bench.bench."""
    sim = BatchSim(load_track("oval"), cfg, n, seed=0, auto_reset=True)
    rng = np.random.default_rng(0)
    sim.reset()
    sim.step(rng.uniform(-1.0, 1.0, (n, 2)))   # warm-up
    decisions, start = 0, time.perf_counter()
    while time.perf_counter() - start < SECONDS:
        sim.step(rng.uniform(-1.0, 1.0, (n, 2)))
        decisions += 1
    return decisions * n / (time.perf_counter() - start)


def main() -> None:
    print(f"{'cars':>5} {'action_repeat':>13} {'car-steps/s':>12} {'sim-s per wall-s':>17}")
    for n in N_CARS:
        for repeat in REPEATS:
            cfg = dataclasses.replace(EnvConfig(), action_repeat=repeat)
            rate = measure(cfg, n)
            print(f"{n:>5} {repeat:>13} {rate:>12,.0f} {rate * repeat * cfg.dt:>17,.0f}")


if __name__ == "__main__":
    main()
