"""Tests for REINFORCE (racer/algos/reinforce.py)."""

import numpy as np
import torch

from racer.algos.reinforce import EXTRA_COLUMNS, ReinforceConfig, reinforce_loss, train_reinforce
from racer.cli import build_parser
from racer.config import EnvConfig
from racer.nets_torch import GaussianActor
from racer.policy import load_policy
from racer.runlog import RunDir


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
    assert after - before >= 0.2


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
