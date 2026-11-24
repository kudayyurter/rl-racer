"""Tests for A2C (racer/algos/a2c.py)."""

import numpy as np

from racer.algos.a2c import (EXTRA_COLUMNS, A2CConfig, add_truncation_bootstrap, nstep_returns,
                             train_a2c)
from racer.cli import build_parser
from racer.config import EnvConfig
from racer.policy import load_policy
from racer.runlog import RunDir


def test_nstep_returns_no_done():
    r = np.ones((3, 1))
    out = nstep_returns(r, np.zeros((3, 1), bool), np.array([0.5]), 0.9)
    assert np.allclose(out, [[3.0745], [2.305], [1.45]])


def test_nstep_returns_done_cuts_bootstrap():
    r = np.ones((3, 1))
    dones = np.array([[False], [True], [False]])
    out = nstep_returns(r, dones, np.array([0.5]), 0.9)
    assert np.allclose(out, [[1.9], [1.0], [1.45]])


def test_truncation_bootstrap():
    r = np.ones((3, 1))
    truncated = np.array([[False], [False], [True]])
    final_values = np.array([[0.0], [0.0], [2.0]])
    out = add_truncation_bootstrap(r, truncated, final_values, 0.9)
    assert np.allclose(out, [[1.0], [1.0], [1.0 + 1.8]])


def test_a2c_smoke(tmp_path):
    cfg = A2CConfig(track="oval", n_cars=4, n_steps=8, hidden=(8,), env_steps_budget=640,
                    eval_every=2)
    run = RunDir.create("a2c", cfg.seed, root=tmp_path, extra_columns=EXTRA_COLUMNS)
    train_a2c(cfg, EnvConfig(max_episode_s=1.0), run)
    rows = run.read_metrics()
    assert len(rows) == 20 and rows[-1]["env_steps"] >= 640
    assert "explained_variance" in rows[0] and np.isfinite(rows[-1]["value_loss"])
    assert not np.isnan(rows[-1]["eval_laps"])
    policy = load_policy(run.path / "checkpoints" / "latest.pt")
    assert policy.act(np.zeros((1, 12))).shape == (1, 2)


def test_cli_knows_a2c():
    assert build_parser().parse_args(["train", "a2c", "--set", "track=oval"]).method == "a2c"
