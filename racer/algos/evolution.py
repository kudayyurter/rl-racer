"""Neuroevolution: learn to drive by survival of the fittest, with no gradients.

The algorithm, one *generation* at a time:

  1. Every car in the population drives one full episode, each with its own network.
  2. A genome's fitness is the total reward its car collected.
  3. The best ``elite`` genomes survive unchanged (so the best score can never drop).
  4. The rest of the next population are children: pick a random elite, copy it, and add
     Gaussian noise to every weight (mutation).
  Repeat. Nothing here knows *why* a genome is good; selection pressure alone improves it.

Because the simulator runs all cars as one numpy batch, a generation costs about as much as a
single episode of a big batch. Training stops once the cumulative number of car-decisions
(``env_steps``) reaches the budget; the generation in progress is always finished.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np

from racer.config import EnvConfig
from racer.evaluate import evaluate
from racer.nets_np import forward_population, genome_size, init_population, unflatten
from racer.policy import register_loader
from racer.runlog import RunDir
from racer.sim import BatchSim, obs_dim
from racer.track import load_track

EXTRA_COLUMNS = ("best_fitness",)  # extra metrics.csv column: best genome's return this generation


@dataclass
class EvolutionConfig:
    env_steps_budget: int           # stop after this many car-decisions
    track: str = "twisty"
    population: int = 128
    elite: int = 16                 # survivors per generation
    mutation_std: float = 0.1       # noise added to each weight of a child
    hidden: tuple[int, ...] = (32,)
    eval_every: int = 5             # generations between evaluations + checkpoints
    seed: int = 0


class EvolutionPolicy:
    """One genome as a Policy: what the viewer and ``evaluate`` drive."""

    def __init__(self, genome: np.ndarray, sizes: tuple[int, ...]):
        self.genome, self.sizes = np.asarray(genome, dtype=float), tuple(sizes)
        self._layers = unflatten(self.genome[None, :], self.sizes)  # leading axis of 1

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic actions for (N, D) observations: the network output, in [-1, 1]."""
        # Same network as training's forward_population, but computed with `x @ w` where training
        # uses einsum. The two sum the same products in a different order, so the actions can
        # differ in the last few bits, and over a long episode an evaluation can end differently
        # from the training episode of the same genome. Left as is on purpose: lesson 04
        # ("One generation, line by line", step 3) explains and measures it.
        x = np.asarray(obs, dtype=float)
        for w, b in self._layers:
            x = np.tanh(x @ w[0] + b[0])
        return x

    def activations(self, obs: np.ndarray) -> list[np.ndarray]:
        """Every layer's output, input first, for the viewer's network panel (`x @ w`, as act)."""
        x = np.asarray(obs, dtype=float)
        acts = [x]
        for w, b in self._layers:
            x = np.tanh(x @ w[0] + b[0])
            acts.append(x)
        return acts

    def layer_weights(self) -> list[np.ndarray]:
        return [w[0] for w, _ in self._layers]


def _load_policy(ckpt: dict) -> EvolutionPolicy:
    return EvolutionPolicy(ckpt["weights"], tuple(ckpt["arch"]["sizes"]))


register_loader("evolution", _load_policy)


def run_generation(sim: BatchSim, genomes: np.ndarray, sizes: tuple[int, ...]):
    """Drive one episode with every car using its own genome.

    Returns (fitness (P,), env_steps): total reward per car, and how many car-decisions were
    actually simulated (a car that has crashed is frozen and does not count).
    """
    obs = sim.reset()
    fitness = np.zeros(len(genomes))
    env_steps = 0
    while sim.alive.any():
        env_steps += int(sim.alive.sum())
        result = sim.step(forward_population(genomes, sizes, obs))
        fitness += result.reward  # frozen cars earn 0, so this is each car's episode return
        obs = result.obs
    return fitness, env_steps


def next_generation(rng: np.random.Generator, genomes: np.ndarray, fitness: np.ndarray,
                    n_elite: int, mutation_std: float) -> np.ndarray:
    """Selection + mutation: keep the elite unchanged, fill the rest with mutated elite copies."""
    pop = len(genomes)
    elites = genomes[np.argsort(fitness)[::-1][:n_elite]]  # best first
    parents = elites[rng.integers(n_elite, size=pop - n_elite)]
    children = parents + rng.normal(0.0, mutation_std, size=parents.shape)
    return np.concatenate([elites, children])


CONFIG_CLASS = EvolutionConfig  # names the CLI looks up for every method (see racer/cli.py)


def train_evolution(cfg: EvolutionConfig, env_cfg: EnvConfig, run: RunDir) -> None:
    if not 1 <= cfg.elite <= cfg.population:
        raise ValueError("need 1 <= elite <= population")
    rng = np.random.default_rng(cfg.seed)  # drives both initialization and mutation
    track = load_track(cfg.track)
    sizes = (obs_dim(env_cfg), *cfg.hidden, 2)  # 2 outputs: [steer, throttle]
    sim = BatchSim(track, env_cfg, n_cars=cfg.population, seed=cfg.seed, auto_reset=False)
    genomes = init_population(rng, cfg.population, sizes)
    run.save_config({"method": "evolution", "seed": cfg.seed,
                     "train": asdict(cfg), "env": asdict(env_cfg)})
    print(f"evolution on {track.name}: population {cfg.population}, elite {cfg.elite}, "
          f"network {sizes} ({genome_size(sizes)} weights)")

    start, env_steps, generation = time.time(), 0, 0
    while env_steps < cfg.env_steps_budget:
        generation += 1
        fitness, steps = run_generation(sim, genomes, sizes)
        env_steps += steps
        best = genomes[int(np.argmax(fitness))].copy()

        row = {"iteration": generation, "env_steps": env_steps,
               "wall_time_s": time.time() - start, "train_return_mean": float(fitness.mean()),
               "eval_return": np.nan, "eval_progress_m": np.nan, "eval_laps": np.nan,
               "eval_best_lap_s": np.nan}
        if "best_fitness" in run.columns:
            row["best_fitness"] = float(fitness.max())
        line = (f"gen {generation:4d}  steps {env_steps:>9,}  "
                f"best {fitness.max():8.2f}  mean {fitness.mean():8.2f}")

        # Evaluate (and checkpoint) periodically, and always on the final generation.
        if generation % cfg.eval_every == 0 or env_steps >= cfg.env_steps_budget:
            policy = EvolutionPolicy(best, sizes)
            res = evaluate(policy, track, env_cfg)
            row.update(eval_return=res.ret, eval_progress_m=res.progress_m,
                       eval_laps=res.laps, eval_best_lap_s=res.best_lap_s)
            run.save_checkpoint({"method": "evolution", "iteration": generation,
                                 "env_steps": env_steps, "obs_dim": sizes[0],
                                 "arch": {"sizes": list(sizes)}, "weights": best,
                                 "env_config": asdict(env_cfg)}, generation)
            if res.laps > 0:
                run.save_ghost(generation, res.lap_t, res.lap_poses, res.best_lap_s)
            line += f"  | eval laps {res.laps}  progress {res.progress_m:6.1f} m"
            if res.laps > 0:
                line += f"  best lap {res.best_lap_s:.2f} s"
        run.log(row)
        print(line, flush=True)

        genomes = next_generation(rng, genomes, fitness, cfg.elite, cfg.mutation_std)


train = train_evolution
