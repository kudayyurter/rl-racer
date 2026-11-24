"""Tests for the numpy population networks (racer/nets_np.py)."""

import numpy as np

from racer.nets_np import forward_population, genome_size, init_population

SIZES = (5, 4, 2)


def test_genome_size():
    assert genome_size(SIZES) == 5 * 4 + 4 + 4 * 2 + 2


def test_init_population_shape_and_zero_biases():
    g = init_population(np.random.default_rng(0), 7, SIZES)
    assert g.shape == (7, genome_size(SIZES))
    assert np.all(g[:, 20:24] == 0) and np.all(g[:, -2:] == 0)
    assert g[:, :20].std() > 0


def test_forward_matches_manual_single():
    g = init_population(np.random.default_rng(1), 1, SIZES)
    x = np.random.default_rng(2).normal(size=(1, 5))
    w1, b1 = g[0, :20].reshape(5, 4), g[0, 20:24]
    w2, b2 = g[0, 24:32].reshape(4, 2), g[0, 32:34]
    expected = np.tanh(np.tanh(x @ w1 + b1) @ w2 + b2)
    out, acts = forward_population(g, SIZES, x, return_activations=True)
    np.testing.assert_allclose(out, expected)
    assert [a.shape for a in acts] == [(1, 5), (1, 4), (1, 2)]


def test_population_members_independent():
    rng = np.random.default_rng(3)
    g = init_population(rng, 3, SIZES)
    x = rng.normal(size=(3, 5))
    before = forward_population(g, SIZES, x)
    g[1] += 1.0
    after = forward_population(g, SIZES, x)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[2], after[2])
    assert not np.allclose(before[1], after[1])
