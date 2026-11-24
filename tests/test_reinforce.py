"""Tests for REINFORCE (racer/algos/reinforce.py)."""

import numpy as np
import torch

from racer.algos.reinforce import (EXTRA_COLUMNS, ReinforceConfig, collect_batch, reinforce_loss,
                                   train_reinforce)
from racer.cli import build_parser
from racer.config import EnvConfig
from racer.nets_torch import GaussianActor
from racer.policy import load_policy
from racer.runlog import RunDir
from racer.sim import BatchSim, obs_dim
from racer.track import load_track


def test_reinforce_gradient_direction():
    """One-step bandit: reward = -(a - 0.5)^2 for steer. Learning must move the mean to 0.5."""
    torch.manual_seed(0)
    actor = GaussianActor(1, 2, (8,))
    obs = torch.ones(256, 1)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    before = actor.mean(obs[:1])[0, 0].item()
    for _ in range(200):
        action = actor.dist(obs).sample()
        reward = -(action[:, 0] - 0.5) ** 2
        loss = reinforce_loss(actor.dist(obs).log_prob(action).sum(-1), reward)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = actor.mean(obs[:1])[0, 0].item()
    assert before < 0.3, "bandit setup: the mean should start well below the 0.5 optimum"
    assert abs(after - 0.5) < 0.1


def test_reinforce_smoke(tmp_path):
    cfg = ReinforceConfig(track="oval", n_cars=4, episodes_per_batch=4, hidden=(8,),
                          env_steps_budget=2000, eval_every=2)
    run = RunDir.create("reinforce", cfg.seed, root=tmp_path, extra_columns=EXTRA_COLUMNS)
    train_reinforce(cfg, EnvConfig(max_episode_s=2.0), run)
    rows = run.read_metrics()
    assert len(rows) >= 2
    assert rows[-1]["env_steps"] >= 2000 and rows[-1]["episode_len_mean"] > 0
    assert not np.isnan(rows[-1]["eval_laps"]) and "log_std_mean" in rows[0]
    policy = load_policy(run.path / "checkpoints" / "latest.pt")
    assert policy.act(np.zeros((1, 12))).shape == (1, 2)
    assert [a.shape for a in policy.activations(np.zeros((1, 12)))] == [(1, 12), (1, 8), (1, 2)]


def test_cli_knows_reinforce():
    args = build_parser().parse_args(["train", "reinforce", "--set", "track=oval"])
    assert args.method == "reinforce"


def test_every_episode_starts_at_a_spawn():
    """Each batch must hold whole episodes: no tails of episodes begun in the previous batch."""
    torch.manual_seed(0)
    track = load_track("oval")
    sim = BatchSim(track, EnvConfig(), n_cars=4, seed=0, auto_reset=True)
    actor = GaussianActor(obs_dim(EnvConfig()), 2, (8,))
    spawn_obs = sim.reset()[0]  # random_start is off, so every spawn gives this observation
    for _ in range(3):
        episodes, _ = collect_batch(sim, actor, 4)
        for ep_obs, _, _ in episodes:
            assert np.allclose(ep_obs[0], spawn_obs)
