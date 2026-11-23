"""Tests for the vectorized ray/segment geometry used by sensors and crash detection."""

import warnings

import numpy as np
import pytest

from racer.geometry import any_crossing, ray_distances


def test_ray_hits_wall_ahead():
    d = ray_distances(np.array([[0., 0.]]), np.array([[[1., 0.]]]),
                      np.array([[[5., -1.]]]), np.array([[[5., 1.]]]), 60.0)
    assert d.shape == (1, 1) and d[0, 0] == pytest.approx(5.0)


def test_ray_miss_returns_max_range():  # wall behind the ray
    d = ray_distances(np.array([[0., 0.]]), np.array([[[1., 0.]]]),
                      np.array([[[-5., -1.]]]), np.array([[[-5., 1.]]]), 60.0)
    assert d[0, 0] == 60.0


def test_ray_beyond_max_range_returns_max_range():
    d = ray_distances(np.array([[0., 0.]]), np.array([[[1., 0.]]]),
                      np.array([[[100., -1.]]]), np.array([[[100., 1.]]]), 60.0)
    assert d[0, 0] == 60.0


def test_ray_passing_beside_segment_end_misses():
    d = ray_distances(np.array([[0., 0.]]), np.array([[[1., 0.]]]),
                      np.array([[[5., 1.]]]), np.array([[[5., 3.]]]), 60.0)
    assert d[0, 0] == 60.0


def test_nearest_of_two_walls():
    d = ray_distances(np.array([[0., 0.]]), np.array([[[1., 0.]]]),
                      np.array([[[7., -1.], [3., -1.]]]), np.array([[[7., 1.], [3., 1.]]]), 60.0)
    assert d[0, 0] == pytest.approx(3.0)


def test_parallel_and_collinear_rays_are_finite():
    # parallel (wall along y=1) and collinear (wall on the ray's own line)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        d = ray_distances(np.zeros((1, 2)), np.array([[[1., 0.], [1., 0.]]]),
                          np.array([[[0., 1.], [2., 0.]]]), np.array([[[9., 1.], [9., 0.]]]), 60.0)
    assert np.all(np.isfinite(d))
    # parallel wall is never hit; the collinear case may report its near end (2.0) or max range
    assert d[0, 0] == 60.0
    assert d[0, 1] in (pytest.approx(2.0), 60.0)


def test_multiple_cars_and_rays_are_independent():
    origins = np.array([[0., 0.], [0., 10.]])
    dirs = np.array([[[1., 0.], [0., 1.]], [[1., 0.], [-1., 0.]]])
    seg_a = np.array([[[5., -1.]], [[4., 9.]]])
    seg_b = np.array([[[5., 20.]], [[4., 11.]]])
    d = ray_distances(origins, dirs, seg_a, seg_b, 60.0)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(5.0)
    assert d[0, 1] == 60.0   # straight up from (0,0) never meets x=5
    assert d[1, 0] == pytest.approx(4.0)
    assert d[1, 1] == 60.0   # facing away from the wall


def test_crossing_detected_and_not():
    p1 = np.array([[[0., 0.]], [[0., 0.]]])
    p2 = np.array([[[2., 2.]], [[2., 2.]]])
    q1 = np.array([[[0., 2.]], [[0., 5.]]])
    q2 = np.array([[[2., 0.]], [[2., 5.]]])
    # car 0: an X shape; car 1: parallel segments far apart
    assert list(any_crossing(p1, p2, q1, q2)) == [True, False]


def test_touching_endpoints_count_as_crossing():
    p1 = np.array([[[0., 0.]]]); p2 = np.array([[[1., 0.]]])
    q1 = np.array([[[1., 0.]]]); q2 = np.array([[[1., 5.]]])
    assert any_crossing(p1, p2, q1, q2)[0]


def test_crossing_any_pair_among_many():
    p1 = np.array([[[0., 0.], [10., 10.]]]); p2 = np.array([[[1., 0.], [12., 10.]]])
    q1 = np.array([[[50., 50.], [11., 9.]]]); q2 = np.array([[[51., 51.], [11., 11.]]])
    assert any_crossing(p1, p2, q1, q2)[0]


def _one(a, b):
    """Wrap a single segment (a->b) as an (N=1, count=1, 2) endpoint pair."""
    return np.array([[a]], dtype=float), np.array([[b]], dtype=float)


def test_lines_intersect_but_p_stops_short():
    # Infinite lines meet at (2, 0); p ends at x=1, so t=2 (out of range), u=0.5.
    p1, p2 = _one((0, 0), (1, 0))
    q1, q2 = _one((2, -1), (2, 1))
    assert not any_crossing(p1, p2, q1, q2)[0]


def test_lines_intersect_but_p_starts_after_crossing():
    # Crossing point is behind p's start: t=-0.5 (out of range), u=0.5.
    p1, p2 = _one((1, 0), (3, 0))
    q1, q2 = _one((0, -1), (0, 1))
    assert not any_crossing(p1, p2, q1, q2)[0]


def test_lines_intersect_but_q_stops_short():
    # Lines meet at (1, 0); q spans y 1..3, so u=-0.5 (out of range), t=0.25.
    p1, p2 = _one((0, 0), (4, 0))
    q1, q2 = _one((1, 1), (1, 3))
    assert not any_crossing(p1, p2, q1, q2)[0]


def test_asymmetric_crossing_is_detected():
    # A true crossing at an off-centre parameter pair (t=0.25 along p, u=0.75 along q) at (1, 0),
    # so the result does not depend on both parameters happening to be 0.5.
    p1, p2 = _one((0, 0), (4, 0))
    q1, q2 = _one((1, -3), (1, 1))
    assert any_crossing(p1, p2, q1, q2)[0]


def test_lines_intersect_but_q_ends_before_crossing():
    # Lines meet at (1, 0); q runs y -3..-1 and stops short of it, so u=1.5 (past q's end), t=0.25.
    p1, p2 = _one((0, 0), (4, 0))
    q1, q2 = _one((1, -3), (1, -1))
    assert not any_crossing(p1, p2, q1, q2)[0]
