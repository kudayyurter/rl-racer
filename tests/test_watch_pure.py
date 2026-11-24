"""Window-free tests for the watch viewer: ghost replay, network layout, curves, hot reload.

Nothing here opens a window. The drawing functions are thin wrappers around the pure helpers
tested below (where things go on screen), and ``WatchSession`` is the viewer's logic without
any drawing, so the hot-reload rules can be checked against a real run directory.
"""

import math
import os

import numpy as np
import pytest

from racer.algos.evolution import EvolutionPolicy
from racer.config import EnvConfig
from racer.render.car_draw import ghost_alphas, ghost_pose_at
from racer.render.curve import curve_points
from racer.render.netpanel import input_labels, node_layout, top_edges
from racer.runlog import RunDir
from racer.sim import obs_dim
from racer.watch import WatchSession


# ---------------------------------------------------------------- ghosts

def test_ghost_interpolation():
    t = np.array([0.0, 1.0, 2.0])
    poses = np.array([[0.0, 0.0, 3.1], [2.0, 4.0, -3.1], [4.0, 4.0, -3.1]])

    mid = ghost_pose_at(t, poses, 0.5)
    assert mid[:2] == pytest.approx([1.0, 2.0])
    # 3.1 -> -3.1 is a small turn through +-pi, not a 6.2 rad spin back through 0.
    assert math.cos(mid[2]) == pytest.approx(-1.0, abs=1e-3)

    quarter = ghost_pose_at(t, poses, 0.25)
    assert quarter[2] == pytest.approx(3.1 + 0.25 * (2 * math.pi - 6.2))

    np.testing.assert_allclose(ghost_pose_at(t, poses, 1.0), poses[1])
    np.testing.assert_allclose(ghost_pose_at(t, poses, 99.0), poses[-1])   # holds the end
    np.testing.assert_allclose(ghost_pose_at(t, poses, -1.0), poses[0])    # and the start


def test_ghost_alphas_fade_older_ghosts():
    alphas = ghost_alphas(4)
    assert len(alphas) == 4
    assert all(a < b for a, b in zip(alphas, alphas[1:]))   # oldest first, faintest
    assert 0.0 < alphas[0] and alphas[-1] <= 1.0
    assert ghost_alphas(0) == []
    assert len(ghost_alphas(1)) == 1


# ---------------------------------------------------------------- network panel

def test_node_layout_fits_rect():
    rect = (100.0, 50.0, 300.0, 400.0)
    sizes = [12, 32, 2]
    layout = node_layout(sizes, rect)
    assert [len(layer) for layer in layout] == sizes
    x, y, w, h = rect
    for layer in layout:
        assert layer.shape[1] == 2
        assert np.all(layer[:, 0] >= x) and np.all(layer[:, 0] <= x + w)
        assert np.all(layer[:, 1] >= y) and np.all(layer[:, 1] <= y + h)
        assert np.all(np.diff(layer[:, 1]) > 0)              # nodes top to bottom, no overlap
        assert np.ptp(layer[:, 0]) == 0                       # one column per layer
    columns = [layer[0, 0] for layer in layout]
    assert columns == sorted(columns) and columns[0] == x and columns[-1] == x + w


def test_node_layout_single_layer():
    (layer,) = node_layout([3], (0.0, 0.0, 100.0, 100.0))
    assert np.all(layer[:, 0] == 50.0)


def test_top_edges_k():
    weights = np.random.default_rng(0).normal(size=(12, 32))
    edges = top_edges(weights, 10)
    assert len(edges) == 10
    for i, j, w in edges:
        assert weights[i, j] == w
    kept = sorted(abs(w) for _, _, w in edges)
    assert kept == sorted(np.abs(weights).ravel())[-10:]
    assert len(top_edges(weights, 10_000)) == weights.size    # k larger than the layer


def test_input_labels():
    assert input_labels(12) == [f"ray {k}" for k in range(1, 10)] + ["speed", "steer", "heading"]


# ---------------------------------------------------------------- learning curve

def test_curve_points_skip_nan_and_fit_rect():
    rows = [{"env_steps": 100.0, "eval_best_lap_s": float("nan")},
            {"env_steps": 200.0, "eval_best_lap_s": 30.0},
            {"env_steps": 300.0, "eval_best_lap_s": 28.0}]
    rect = (10.0, 20.0, 200.0, 100.0)
    pts = curve_points(rows, "eval_best_lap_s", rect)
    assert pts.shape == (2, 2)
    x, y, w, h = rect
    assert np.all(pts[:, 0] >= x) and np.all(pts[:, 0] <= x + w)
    assert np.all(pts[:, 1] >= y) and np.all(pts[:, 1] <= y + h)
    assert pts[1, 0] > pts[0, 0]          # later steps further right
    assert pts[1, 1] > pts[0, 1]          # smaller value drawn lower (screen y grows down)
    assert curve_points(rows[:1], "eval_best_lap_s", rect).shape == (0, 2)


# ---------------------------------------------------------------- hot reload

def _run_with_checkpoint(tmp_path, env: dict, iteration: int = 1) -> RunDir:
    run = RunDir.create("evolution", 0, root=tmp_path)
    run.save_config({"method": "evolution", "seed": 0, "train": {"track": "oval"}, "env": env})
    _save_policy(run, iteration)
    return run


def _save_policy(run: RunDir, iteration: int) -> None:
    sizes = (obs_dim(EnvConfig()), 4, 2)
    policy = EvolutionPolicy(np.zeros(sum(i * o + o for i, o in zip(sizes, sizes[1:]))), sizes)
    run.save_checkpoint({"method": "evolution", "iteration": iteration,
                         "env_steps": 1000 * iteration, "arch": {"sizes": list(sizes)},
                         "weights": policy.genome}, iteration)
    # Two saves within one filesystem timestamp tick would look unchanged; push mtime forward.
    latest = run.path / "checkpoints" / "latest.pt"
    stamp = os.stat(latest).st_mtime_ns + iteration * 10**9
    os.utime(latest, ns=(stamp, stamp))


def test_waiting_until_first_checkpoint(tmp_path):
    run = RunDir.create("evolution", 0, root=tmp_path)
    run.save_config({"method": "evolution", "seed": 0, "train": {"track": "oval"},
                     "env": {"max_episode_s": 1.0}})
    session = WatchSession(run.path, reload_s=1.0, n_ghosts=3)
    assert session.policy is None and session.status_lines()   # waiting, but no crash
    session.update(0.1)
    _save_policy(run, 1)
    session.update(1.0)   # the next poll loads it straight away: nothing is driving yet
    assert session.policy is not None and session.checkpoint["iteration"] == 1


def test_new_checkpoint_swaps_in_only_when_episode_ends(tmp_path):
    # A zero-weight network outputs 0 throttle: the car sits still until the 1 s time limit.
    run = _run_with_checkpoint(tmp_path, {"max_episode_s": 1.0})
    session = WatchSession(run.path, reload_s=0.05, n_ghosts=3)
    assert session.checkpoint["iteration"] == 1
    assert session.track.name == "oval"            # from config.json, not a default
    assert session.cfg.max_episode_s == 1.0        # the run's env config, not env.toml

    _save_policy(run, 2)
    session.update(0.1)                            # polls and sees it, mid-episode
    assert session.pending and session.checkpoint["iteration"] == 1
    for _ in range(12):                            # > 1 s of sim time: the episode ends
        session.update(0.1)
    assert not session.pending and session.checkpoint["iteration"] == 2


def test_bad_checkpoint_shows_error_instead_of_crashing(tmp_path):
    run = _run_with_checkpoint(tmp_path, {"max_episode_s": 1.0})
    session = WatchSession(run.path, reload_s=0.05, n_ghosts=3)
    run.save_checkpoint({"method": "nonsense", "iteration": 9}, 9)
    latest = run.path / "checkpoints" / "latest.pt"
    stamp = os.stat(latest).st_mtime_ns + 10**10
    os.utime(latest, ns=(stamp, stamp))
    for _ in range(15):
        session.update(0.1)
    assert session.checkpoint["iteration"] == 1    # still driving the last good policy
    assert "nonsense" in session.error
    assert any("nonsense" in line for line in session.status_lines())
