"""Lesson 04: what the 128 random networks of generation 1 do on twisty.

Rebuilds generation 1 exactly as ``racer train evolution`` does with seed 0 (same RNG, same
initialization, same simulator), drives it, and counts how many cars ran the full 90 s episode,
how many of those barely moved (braking from the start, so they sit on the line until the time
limit), and what share of the generation's env steps those full-length cars used.

Run: uv run python docs/lessons/scripts/first_generation.py
"""

import numpy as np

from racer.config import EnvConfig
from racer.nets_np import forward_population, init_population
from racer.sim import BatchSim, obs_dim
from racer.track import load_track

SEED, POPULATION = 0, 128


def main() -> None:
    cfg = EnvConfig()
    sizes = (obs_dim(cfg), 32, 2)
    genomes = init_population(np.random.default_rng(SEED), POPULATION, sizes)
    sim = BatchSim(load_track("twisty"), cfg, n_cars=POPULATION, seed=SEED, auto_reset=False)
    obs = sim.reset()
    steps, fitness = np.zeros(POPULATION), np.zeros(POPULATION)
    while sim.alive.any():
        steps += sim.alive
        result = sim.step(forward_population(genomes, sizes, obs))
        fitness += result.reward
        obs = result.obs
    full = steps >= steps.max()
    parked = full & (sim.traveled < 5.0)
    print(f"generation 1: {int(steps.sum()):,} env steps, mean fitness {fitness.mean():.2f}")
    print(f"  {full.sum()} of {POPULATION} cars drove the full {int(steps.max())} decisions; "
          f"{parked.sum()} of them moved less than 5 m (fitness {fitness[parked].max():.2f})")
    print(f"  full-length cars used {steps[full].sum() / steps.sum():.0%} of the steps")


if __name__ == "__main__":
    main()
