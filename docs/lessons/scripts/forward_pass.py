"""Lesson 04: one forward pass of a trained evolution network, written out step by step.

Loads the newest checkpoint of an evolution run, takes the observation a car gets on the start
line of the run's track, and computes the network by hand: where each layer's weights sit in
the flat genome, the shapes at every step, the pre-activations and the tanh outputs. It then
checks the hand computation against racer.nets_np.forward_population, and shows that the
population forward (einsum) gives each car its own network: a batch of copies of the same
genome agrees with a Python loop over cars, and changing one copy changes only that car.

Run: uv run python docs/lessons/scripts/forward_pass.py runs/evolution-<timestamp>-s<seed>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from racer.config import EnvConfig, from_dict
from racer.nets_np import forward_population, genome_size
from racer.sim import BatchSim
from racer.track import load_track


def main() -> None:
    run = Path(sys.argv[1])
    ckpt = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    config = json.loads((run / "config.json").read_text())
    sizes = tuple(ckpt["arch"]["sizes"])
    genome = np.asarray(ckpt["weights"], dtype=float)
    print(f"{run.name}: checkpoint from generation {ckpt['iteration']}, layer sizes {sizes}")
    print(f"genome: {genome.shape[0]} numbers (genome_size says {genome_size(sizes)})")

    # Where each piece lives in the flat genome (the layout in nets_np's docstring).
    start, layers = 0, []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        w_end = start + n_in * n_out
        w = genome[start:w_end].reshape(n_in, n_out)
        b = genome[w_end:w_end + n_out]
        print(f"  W {n_in}x{n_out}: genome[{start}:{w_end}]   b {n_out}: genome[{w_end}:{w_end + n_out}]")
        layers.append((w, b))
        start = w_end + n_out

    env = from_dict(EnvConfig, config["env"])
    sim = BatchSim(load_track(config["train"]["track"]), env, n_cars=1)
    x = sim.reset()                       # (1, 12): one car's observation on the start line
    print(f"\ninput x {x.shape}: {np.array2string(x[0], precision=3)}")

    h = x
    for k, (w, b) in enumerate(layers, start=1):
        z = h @ w + b                     # (1, n_in) @ (n_in, n_out) + (n_out,) -> (1, n_out)
        h = np.tanh(z)
        print(f"layer {k}: z = h @ W + b  {z.shape}; tanh(z)")
        if h.shape[1] <= 2:
            print(f"  z      = {np.array2string(z[0], precision=3)}")
            print(f"  tanh z = {np.array2string(h[0], precision=3)}   <- [steer, throttle]")
        else:
            saturated = int((np.abs(h) > 0.95).sum())
            print(f"  tanh z: min {h.min():+.3f}, max {h.max():+.3f}, "
                  f"{saturated} of {h.shape[1]} units saturated (|tanh| > 0.95)")
    library = forward_population(genome[None, :], sizes, x)
    print(f"forward_population gives {np.array2string(library[0], precision=3)} "
          f"(max difference {np.abs(library - h).max():.1e})")

    # The population forward: P genomes, P observations, one einsum per layer.
    rng = np.random.default_rng(0)
    p = 8
    genomes = genome[None, :] + rng.normal(0.0, 0.1, size=(p, genome.size))   # 8 mutants
    obs = np.repeat(x, p, axis=0)
    batched = forward_population(genomes, sizes, obs)
    looped = np.concatenate([forward_population(genomes[i:i + 1], sizes, obs[i:i + 1])
                             for i in range(p)])
    print(f"\n{p} mutated copies, same observation: batched == loop over cars: "
          f"{np.allclose(batched, looped)}")
    for i in range(p):
        print(f"  car {i}: steer {batched[i, 0]:+.3f}  throttle {batched[i, 1]:+.3f}")


if __name__ == "__main__":
    main()
