"""Tests for PPO (racer/algos/ppo.py)."""

import numpy as np
import pytest
import torch

from racer.algos.a2c import nstep_returns
from racer.algos.ppo import EXTRA_COLUMNS, PPOConfig, clipped_surrogate, compute_gae, train_ppo
from racer.cli import build_parser
from racer.config import EnvConfig
from racer.policy import load_policy
from racer.runlog import RunDir


def test_gae_no_done():
    r, v = np.ones((3, 1)), np.full((3, 1), 0.5)
    out = compute_gae(r, v, np.zeros((3, 1), bool), np.array([0.5]), 0.9, 0.8)
    assert np.allclose(out[:, 0], [2.12648, 1.634, 0.95])


def test_gae_done_at_t1():
    r, v = np.ones((3, 1)), np.full((3, 1), 0.5)
    dones = np.array([[False], [True], [False]])
    out = compute_gae(r, v, dones, np.array([0.5]), 0.9, 0.8)
    assert np.allclose(out[:, 0], [1.31, 0.5, 0.95])


def test_gae_lam1_equals_nstep_minus_value():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    dones = rng.random((6, 3)) < 0.2
    last = rng.normal(size=3)
    gae = compute_gae(r, v, dones, last, 0.97, 1.0)
    assert np.allclose(gae, nstep_returns(r, dones, last, 0.97) - v)


def test_clip_blocks_large_ratio():
    logp_new = torch.tensor([np.log(2.0)], requires_grad=True)  # ratio 2.0
    loss = clipped_surrogate(logp_new, torch.zeros(1), torch.ones(1), 0.2)
    loss.backward()
    assert logp_new.grad.item() == 0.0
    assert loss.item() == pytest.approx(-1.2)


def test_clip_passes_gradient_inside_range():
    logp_new = torch.tensor([0.0], requires_grad=True)  # ratio 1.0
    clipped_surrogate(logp_new, torch.zeros(1), torch.ones(1), 0.2).backward()
    assert logp_new.grad.item() == pytest.approx(-1.0)


def test_ppo_smoke(tmp_path):
    cfg = PPOConfig(track="oval", n_cars=4, n_steps=8, epochs=2, minibatch=16, hidden=(8,),
                    env_steps_budget=320, eval_every=2)
    run = RunDir.create("ppo", cfg.seed, root=tmp_path, extra_columns=EXTRA_COLUMNS)
    train_ppo(cfg, EnvConfig(max_episode_s=1.0), run)
    rows = run.read_metrics()
    assert len(rows) == 10 and rows[-1]["env_steps"] >= 320
    for col in ("approx_kl", "clip_frac", "value_loss"):
        assert np.isfinite(rows[-1][col])
    assert not np.isnan(rows[-1]["eval_laps"])
    policy = load_policy(run.path / "checkpoints" / "latest.pt")
    assert policy.act(np.zeros((1, 12))).shape == (1, 2)


def test_random_tracks_not_implemented(tmp_path):
    cfg = PPOConfig(track="oval", tracks="random", env_steps_budget=64)
    run = RunDir.create("ppo", 0, root=tmp_path, extra_columns=EXTRA_COLUMNS)
    with pytest.raises(NotImplementedError):
        train_ppo(cfg, EnvConfig(), run)


def test_cli_knows_ppo():
    assert build_parser().parse_args(["train", "ppo", "--set", "track=oval"]).method == "ppo"
