"""SB3 answer key: it must match our PPO's settings and write the same run-directory format."""

import numpy as np
import torch

from racer.algos import ppo as our_ppo
from racer.algos.sb3_ppo import EXTRA_COLUMNS, SB3Config, make_envs, train_sb3
from racer.cli import build_parser
from racer.config import EnvConfig, load_method_config, load_toml
from racer.policy import load_policy
from racer.runlog import METRIC_COLUMNS, RunDir


def test_sb3_smoke(tmp_path):
    cfg = SB3Config(track="oval", n_envs=4, n_steps=64, batch_size=64, n_epochs=2,
                    env_steps_budget=4096, eval_every_steps=1024)
    run = RunDir.create("sb3", cfg.seed, root=tmp_path, extra_columns=EXTRA_COLUMNS)
    train_sb3(cfg, EnvConfig(max_episode_s=2.0), run)
    rows = run.read_metrics()
    assert len(rows) >= 2 and rows[-1]["env_steps"] >= 4096
    assert all(c in rows[-1] for c in METRIC_COLUMNS)
    # Strictly increasing: chunked learn() continues the run instead of restarting it.
    for prev, cur in zip(rows, rows[1:]):
        assert cur["iteration"] > prev["iteration"] and cur["env_steps"] > prev["env_steps"]
    assert not np.isnan(rows[-1]["eval_laps"])
    policy = load_policy(run.path / "checkpoints" / "latest.pt")
    assert policy.act(np.zeros((1, 12))).shape == (1, 2)
    # The viewer's network panel: input, two hidden layers, output; weights as (in, out).
    acts = policy.activations(np.zeros((1, 12)))
    assert [a.shape[-1] for a in acts] == [12, 64, 64, 2]
    assert [w.shape for w in policy.layer_weights()] == [(12, 64), (64, 64), (64, 2)]
    ckpt = torch.load(run.path / "checkpoints" / "latest.pt", weights_only=False)
    assert ckpt["method"] == "sb3" and (run.path / ckpt["weights"]).exists()


def test_sb3_config_matches_ppo_toml():
    """The comparison is only fair if every shared setting is the same number."""
    ppo = load_toml("configs/ppo.toml")
    sb3 = load_toml("configs/sb3.toml")
    pairs = {"n_envs": "n_cars", "n_steps": "n_steps", "batch_size": "minibatch",
             "n_epochs": "epochs", "gamma": "gamma", "gae_lambda": "lam", "clip_range": "clip",
             "ent_coef": "entropy_coef", "vf_coef": "value_coef",
             "max_grad_norm": "max_grad_norm", "learning_rate": "lr",
             "normalize_advantage": "normalize_adv", "track": "track", "seed": "seed",
             "env_steps_budget": "env_steps_budget"}
    for sb3_key, ppo_key in pairs.items():
        assert sb3[sb3_key] == ppo[ppo_key], sb3_key
    # Same evaluation cadence: PPO evaluates every `eval_every` iterations of n_cars * n_steps.
    assert sb3["eval_every_steps"] == ppo["eval_every"] * ppo["n_cars"] * ppo["n_steps"]
    assert tuple(sb3["net_arch"]) == tuple(ppo["hidden"])
    # The file loads through the strict loader into a config with the same shared numbers.
    cfg = load_method_config(SB3Config, "configs/sb3.toml")
    assert cfg.n_steps == our_ppo.PPOConfig.n_steps and cfg.batch_size == 256


def test_envs_get_different_seeds():
    """Without seed=cfg.seed every env would start from RacerEnv's default seed 0."""
    envs = make_envs(SB3Config(track="oval", n_envs=3, seed=5, env_steps_budget=1), EnvConfig(random_start=True))
    obs = envs.reset()
    assert obs.shape == (3, 12)
    assert len({tuple(row) for row in obs.round(5)}) == 3


def test_cli_knows_sb3():
    assert build_parser().parse_args(["train", "sb3", "--set", "track=oval"]).method == "sb3"
