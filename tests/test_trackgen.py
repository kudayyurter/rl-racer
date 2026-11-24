"""Tests for the random track generator and the validity check (racer/trackgen.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers import circle_points, hairpin_points, wall_self_intersects, walls_cross

from racer.cli import main
from racer.track import build_track, load_track
from racer.trackgen import (HELDOUT_FIRST_SEED, _walls_cross_each_other, heldout_tracks,
                            random_track, track_for_seed, track_is_valid, training_seed)


def test_random_tracks_valid():
    for seed in range(50):
        t = random_track(np.random.default_rng(seed))
        # Independent oracle (the slow pairwise helper), not the function under test.
        assert not wall_self_intersects(t.left) and not wall_self_intersects(t.right)
        assert not walls_cross(t.left, t.right)
        assert 400 <= t.length <= 1400
        assert track_is_valid(t)


def test_seeded_reproducible():
    a = random_track(np.random.default_rng(7))
    b = random_track(np.random.default_rng(7))
    c = random_track(np.random.default_rng(8))
    assert np.array_equal(a.center, b.center)
    assert not (a.center.shape == c.center.shape and np.allclose(a.center, c.center))


def test_heldout_disjoint_from_training_seeds():
    held = heldout_tracks(10)
    assert len(held) == 10
    rng = np.random.default_rng(0)
    seeds = {training_seed(rng, 10) for _ in range(5000)}
    assert not seeds & set(range(HELDOUT_FIRST_SEED, HELDOUT_FIRST_SEED + 10))


def test_training_seed_redraws_inside_heldout_range():
    class Stub:  # first draw lands in the held-out range, second is fine
        draws = iter([HELDOUT_FIRST_SEED + 3, 42])

        def integers(self, lo, hi):
            return next(self.draws)

    assert training_seed(Stub(), 10) == 42


def test_heldout_tracks_are_fixed():
    a, b = heldout_tracks(3), heldout_tracks(3)
    assert all(np.array_equal(x.center, y.center) for x, y in zip(a, b))


def test_random_track_gives_up():
    with pytest.raises(RuntimeError):
        random_track(np.random.default_rng(0), radius=(5.0, 6.0), max_tries=3)


def test_validity_accepts_reasonable_shapes():
    assert track_is_valid(build_track("c", circle_points(80, 24), 12.0))
    assert track_is_valid(build_track("hp", hairpin_points(gap=40.0, straight=120.0), 12.0))


def test_validity_rejects_tight_corner():
    # circle of radius 8 < width/2 + 3 = 9
    assert not track_is_valid(build_track("tight", circle_points(8, 16), 12.0))


def test_left_vs_right_wall_check_sees_what_self_checks_miss():
    # Two same-direction sections side by side, the second shifted so that its right wall
    # pokes through the first's left wall. Each wall alone is a clean closed polyline.
    left = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    right = left + np.array([5.0, 2.0])
    assert not wall_self_intersects(left) and not wall_self_intersects(right)
    assert walls_cross(left, right)
    assert _walls_cross_each_other(SimpleNamespace(left=left, right=right))


def test_validity_rejects_overlapping_parallel_sections():
    # Parallel straights 8 m apart with width 12: the asphalt overlaps (the walls are
    # parallel, so no crossing is found; the proximity and corner checks reject it).
    t = build_track("hp", hairpin_points(gap=8.0, straight=120.0), 12.0)
    assert not track_is_valid(t)


def test_validity_rejects_close_sections_without_wall_crossing():
    # Straights 14 m apart: walls (2 m gap) cross nothing, but the asphalt of the two
    # sections touches nearly (centerlines closer than width + 4 = 16 m).
    t = build_track("hp", hairpin_points(gap=14.0, straight=120.0), 12.0)
    assert not walls_cross(t.left, t.right)
    assert not track_is_valid(t)


def test_gen_track_round_trip(tmp_path):
    out = tmp_path / "rand_5.json"
    assert main(["gen-track", "--seed", "5", "--out", str(out)]) == 0
    loaded = load_track(str(out))
    original = track_for_seed(5)
    assert loaded.length == pytest.approx(original.length, rel=1e-6)
    assert len(loaded.center) == len(original.center)
