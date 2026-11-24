"""Tests for the track model: spline sampling, walls, windowed projection, pose lookup."""

import numpy as np
import pytest
from helpers import circle_points, hairpin_points, wall_self_intersects, walls_cross

from racer.track import build_track, catmull_rom_closed, load_track


def test_circle_track_length():
    t = build_track("c", circle_points(50, 24), 12.0)
    assert t.length == pytest.approx(2 * np.pi * 50, rel=0.01)
    assert np.allclose(np.linalg.norm(t.left - t.center, axis=1), 6.0, atol=1e-6)
    assert np.diff(t.cum_len).std() < 0.05 * t.spacing  # even spacing


def test_tangent_follows_point_order():
    t = build_track("c", circle_points(50, 24), 12.0)  # CCW points
    assert np.allclose(np.linalg.norm(t.tangent, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(t.normal, axis=1), 1.0)
    # left wall is the inner wall for a CCW circle, right wall the outer one
    assert np.all(np.linalg.norm(t.left, axis=1) < np.linalg.norm(t.center, axis=1))
    assert np.all(np.linalg.norm(t.right, axis=1) > np.linalg.norm(t.center, axis=1))
    # CCW: tangent has positive angular momentum about the origin
    assert np.all(t.center[:, 0] * t.tangent[:, 1] - t.center[:, 1] * t.tangent[:, 0] > 0)


def test_catmull_rom_passes_through_control_points():
    pts = circle_points(50, 12)
    out = catmull_rom_closed(pts, 600)
    assert out.shape == (600, 2)
    for p in pts:
        assert np.linalg.norm(out - p, axis=1).min() < 1.0


def test_start_index_converts_to_nearest_sample():
    pts = circle_points(50, 24)
    t = build_track("c", pts, 12.0, start_index=6)  # control point 6 is at 90 degrees
    assert np.linalg.norm(t.center[t.start_index] - pts[6]) < t.spacing


def test_project_on_centerline_round_trip():
    t = load_track("oval")
    s = np.array([0.0, 100.0, t.length - 1.0])
    pos, _ = t.pose_at(s)
    idx = (s / t.spacing).astype(int)
    _, s2 = t.project(pos, idx)
    assert np.allclose(s2, s, atol=0.05)


def test_project_is_continuous_between_samples():
    t = build_track("c", circle_points(50, 24), 12.0)
    s = np.array([10.3 * t.spacing, 10.5 * t.spacing, 10.7 * t.spacing])
    pos, _ = t.pose_at(s)
    _, s2 = t.project(pos, (s / t.spacing).astype(int))
    assert np.allclose(s2, s, atol=0.05)


def test_project_wraps_across_start_line():
    t = load_track("oval")
    pos, _ = t.pose_at(np.array([t.length - 0.5, t.length + 0.5]))
    _, s = t.project(pos, np.array([t.center.shape[0] - 1, 0]))
    assert 0.0 <= s.min() and s.max() < t.length
    assert s[0] == pytest.approx(t.length - 0.5, abs=0.05)
    assert s[1] == pytest.approx(0.5, abs=0.05)


def test_pose_at_interpolates_and_wraps():
    t = build_track("c", circle_points(50, 24), 12.0)
    pos, heading = t.pose_at(np.array([0.5 * t.spacing, t.length + 0.5 * t.spacing]))
    mid = 0.5 * (t.center[0] + t.center[1])
    assert np.allclose(pos[0], mid, atol=1e-6) and np.allclose(pos[1], mid, atol=1e-6)
    # CCW circle at angle ~0: heading is about +90 degrees
    assert heading[0] == pytest.approx(np.pi / 2, abs=0.1)


def test_wall_window_shapes_and_wrapping():
    t = build_track("c", circle_points(50, 24), 12.0)
    m = len(t.center)
    a, b = t.wall_window(np.array([0, m - 1]), half=3)
    k = 2 * (2 * 3 + 1)
    assert a.shape == (2, k, 2) and b.shape == (2, k, 2)
    # segment i of the left wall runs from sample i to i+1; first window entry is idx-half
    assert np.allclose(a[0, 0], t.left[(0 - 3) % m]) and np.allclose(b[0, 0], t.left[(0 - 2) % m])
    assert np.allclose(b[1, 3], t.left[0])  # last sample's segment closes the loop
    assert np.allclose(a[0, 7], t.right[(0 - 3) % m])  # right wall follows the left wall


def test_projection_does_not_jump_hairpin():
    # U-turn track: two parallel straights 14 m apart, width 12 (walls 2 m apart)
    t = build_track("hp", hairpin_points(gap=14.0, straight=120.0), 12.0)
    s = np.arange(0.0, 100.0, 3.3)
    pos, _ = t.pose_at(s)
    pos = pos + t.normal[(s / t.spacing).astype(int)] * 4.0  # drive near the inner edge
    hint = np.array([0])
    prev = 0.0
    for p in pos:
        hint, s_now = t.project(p[None], hint)
        # Wrap-aware: the first sample sits right at the start line, where s may read as -0.3
        # i.e. length - 0.3, which is a tiny step, not a jump.
        step = (s_now[0] - prev + t.length / 2) % t.length - t.length / 2
        assert abs(step) < 3.3 * 1.5
        prev = s_now[0]


def test_projection_search_is_windowed():
    # A point at y=8 is nearer the far straight's centerline (y=14, 6 m away) than its own
    # (y=0, 8 m away), but the hint says the car is on the near straight. The windowed search
    # must stay near the hint; a global search would jump to the far straight.
    t = build_track("hp", hairpin_points(gap=14.0, straight=120.0), 12.0)
    m = len(t.center)
    half = 10
    hint = np.array([int(60.0 / t.spacing)])
    point = np.array([[60.0, 8.0]])

    def circ_dist(a, b):
        d = abs(int(a) - int(b)) % m
        return min(d, m - d)

    idx, _ = t.project(point, hint, half=half)
    global_idx, _ = t.project(point, hint, half=m // 2)
    assert circ_dist(idx[0], hint[0]) <= half
    assert circ_dist(global_idx[0], hint[0]) > half  # proves the test discriminates


def test_bounds_include_walls():
    t = build_track("c", circle_points(50, 24), 12.0)
    xmin, ymin, xmax, ymax = t.bounds()
    assert xmax == pytest.approx(56, abs=0.5) and xmin == pytest.approx(-56, abs=0.5)
    assert ymax == pytest.approx(56, abs=0.5) and ymin == pytest.approx(-56, abs=0.5)


def test_load_track_by_path(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"name": "x", "width": 12, "start_index": 0, '
                 '"points": [[0,0],[50,0],[60,40],[20,60],[-20,40],[-30,0]]}')
    assert load_track(str(p)).name == "x"


def test_shipped_tracks_valid():
    for name in ("oval", "twisty"):
        t = load_track(name)
        # walls never self-intersect (check every segment pair of each wall, excluding neighbors)
        assert not wall_self_intersects(t.left) and not wall_self_intersects(t.right)
        # ...and the two walls never cross each other (sections running side by side)
        assert not walls_cross(t.left, t.right)
    assert 900 <= load_track("twisty").length <= 1400
