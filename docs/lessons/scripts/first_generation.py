"""Lesson 04: what the random networks of generation 1 do (128 on twisty with the defaults).

Rebuilds generation 1 exactly as ``racer train evolution`` does with seed 0 and the default
configs (configs/evolution.toml for track, population and hidden sizes, configs/env.toml for the
environment; same RNG, same initialization, same simulator), drives it, and counts how many cars ran the full 90 s episode,
how many of those barely moved (braking from the start, so they sit on the line until the time
limit), and what share of the generation's env steps those full-length cars used.

Run: uv run python docs/lessons/scripts/first_generation.py
"""

import numpy as np

from racer.algos.evolution import EvolutionConfig
from racer.config import load_env_config, load_method_config
from racer.nets_np import forward_population, init_population
from racer.sim import BatchSim, obs_dim
from racer.track import load_track

SEED = 0


def main() -> None:
    train = load_method_config(EvolutionConfig, "configs/evolution.toml")
    cfg = load_env_config("configs/env.toml")
    sizes = (obs_dim(cfg), *train.hidden, 2)   # as in train_evolution
    pop = train.population
    genomes = init_population(np.random.default_rng(SEED), pop, sizes)
    sim = BatchSim(load_track(train.track), cfg, n_cars=pop, seed=SEED, auto_reset=False)
    obs = sim.reset()
    steps, fitness = np.zeros(pop), np.zeros(pop)
    while sim.alive.any():
        steps += sim.alive
        result = sim.step(forward_population(genomes, sizes, obs))
        fitness += result.reward
        obs = result.obs
    full = steps >= steps.max()
    parked = full & (sim.traveled < 5.0)
    print(f"generation 1: {int(steps.sum()):,} env steps, mean fitness {fitness.mean():.2f}")
    print(f"  {full.sum()} of {pop} cars drove the full {int(steps.max())} decisions; "
          f"{parked.sum()} of them moved less than 5 m (fitness {fitness[parked].max():.2f})")
    print(f"  full-length cars used {steps[full].sum() / steps.sum():.0%} of the steps")


if __name__ == "__main__":
    main()
