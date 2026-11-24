"""Tiny multilayer perceptrons in plain numpy, for a whole population at once.

Why this exists: neuroevolution has no gradients, so it does not need a deep-learning
framework. A network is just a flat vector of numbers (a *genome*), which makes the genetic
operators trivial: copy a vector, add noise to a vector. This module turns genomes into
networks and runs P different networks (one per car) on P observations in one numpy call.

Genome layout, for layer sizes (in, h1, ..., out): for each layer in order, the weight
matrix W of shape (n_in, n_out) flattened row-major, then the bias b of shape (n_out,).
Every layer, including the output, uses tanh, so outputs land in [-1, 1] and map straight
to the car's [steer, throttle] actions.
"""

import numpy as np


def _layer_shapes(sizes: tuple[int, ...]) -> list[tuple[int, int]]:
    return list(zip(sizes[:-1], sizes[1:]))


def genome_size(sizes: tuple[int, ...]) -> int:
    """Number of parameters (weights and biases) in a network with these layer sizes."""
    return sum(n_in * n_out + n_out for n_in, n_out in _layer_shapes(sizes))


def init_population(rng: np.random.Generator, pop: int, sizes: tuple[int, ...]) -> np.ndarray:
    """(pop, genome_size) random genomes: Xavier-scaled normal weights, zero biases.

    Xavier scaling (std = sqrt(2 / (n_in + n_out))) keeps tanh units out of saturation at the
    start, so different genomes produce different, non-constant behavior.
    """
    genomes = np.zeros((pop, genome_size(sizes)))
    start = 0
    for n_in, n_out in _layer_shapes(sizes):
        std = np.sqrt(2.0 / (n_in + n_out))
        end = start + n_in * n_out
        genomes[:, start:end] = rng.normal(0.0, std, size=(pop, n_in * n_out))
        start = end + n_out  # biases stay zero
    return genomes


def unflatten(genomes: np.ndarray, sizes: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split genomes (P, G) into per-layer views (W (P, in, out), b (P, out)). No copying."""
    p = genomes.shape[0]
    layers, start = [], 0
    for n_in, n_out in _layer_shapes(sizes):
        w_end = start + n_in * n_out
        w = genomes[:, start:w_end].reshape(p, n_in, n_out)
        b = genomes[:, w_end:w_end + n_out]
        layers.append((w, b))
        start = w_end + n_out
    return layers


def forward_population(genomes: np.ndarray, sizes: tuple[int, ...], x: np.ndarray,
                       return_activations: bool = False):
    """Run network p on input row p, for every p at once. x: (P, in) -> (P, out).

    ``einsum('pi,pio->po')`` is a batched matrix-vector product where each row of x meets its
    own weight matrix. With ``return_activations`` also returns [input, hidden..., output].
    """
    activations = [x]
    for w, b in unflatten(genomes, sizes):
        x = np.tanh(np.einsum("pi,pio->po", x, w) + b)
        activations.append(x)
    return (x, activations) if return_activations else x
