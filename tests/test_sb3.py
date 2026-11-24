"""SB3 answer key: it must match our PPO's settings and write the same run-directory format."""

import numpy as np
import torch
from stable_baselines3 import PPO

from racer.algos import ppo as our_ppo
from racer.algos.sb3_ppo import EXTRA_COLUMNS, BatchSimVecEnv, SB3Config, make_envs, train_sb3
from racer.cli import build_parser
from racer.config import EnvConfig, load_method_config, load_toml
from racer.gym_env import RacerEnv
from racer.policy import load_policy
from racer.runlog import METRIC_COLUMNS, RunDir
from racer.sim import BatchSim
from racer.track import load_track


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


def test_cars_get_different_starts():
    """One sim, one RNG: with random starts the cars must not all spawn at the same place."""
    envs = make_envs(SB3Config(track="oval", n_envs=3, seed=5, env_steps_budget=1), EnvConfig(random_start=True))
    obs = envs.reset()
    assert obs.shape == (3, 12) and obs.dtype == np.float32
    assert len({tuple(row) for row in obs.round(5)}) == 3


def _actions(n_steps: int, seed: int = 1) -> np.ndarray:
    """Mostly-straight driving (so time limits hit) with bursts of random steering (crashes)."""
    rng = np.random.default_rng(seed)
    acts = rng.uniform(-1, 1, (n_steps, 2))
    acts[:, 1] = 1.0             # full throttle: crashes come quickly
    acts[:100, 0] = 0.0          # first 100 decisions: straight, so time limits end episodes
    return acts


def test_vec_env_matches_racer_env_step_by_step():
    """BatchSimVecEnv with one car is RacerEnv plus auto-reset: same rewards and dones."""
    cfg = EnvConfig(max_episode_s=4.0)
    acts = _actions(300)
    env = RacerEnv("oval", cfg)
    vec = BatchSimVecEnv(load_track("oval"), cfg, n_envs=1, seed=0)
    env.reset(seed=0)
    vec.reset()
    n_trunc = n_crash = 0
    for a in acts:
        _, r, term, trunc, _ = env.step(a)
        vec.step_async(a[None])
        _, vr, vdone, vinfo = vec.step_wait()
        assert vr[0] == np.float32(r) and vr.dtype == np.float32
        assert bool(vdone[0]) == (term or trunc)
        assert vinfo[0]["TimeLimit.truncated"] == (trunc and not term)
        n_trunc += trunc
        n_crash += term
        if term or trunc:
            env.reset()
    assert n_trunc >= 1 and n_crash >= 1


def test_vec_env_done_step_semantics():
    cfg = EnvConfig(max_episode_s=2.0)
    track = load_track("oval")
    vec = BatchSimVecEnv(track, cfg, n_envs=4, seed=3)
    # A mirror sim with the same seed and actions shows what the wrapper should report.
    mirror = BatchSim(track, cfg, n_cars=4, seed=3, auto_reset=True)
    vec.reset()
    mirror.reset()
    rng = np.random.default_rng(0)
    returns, seen_trunc, seen_crash = np.zeros(4), False, False
    for step in range(400):
        a = rng.uniform(-1, 1, (4, 2))
        a[:, 1] = 1.0
        if step < 60:
            a[:, 0] = 0.0
        vec.step_async(a)
        obs, rew, dones, infos = vec.step_wait()
        ref = mirror.step(a)
        returns += rew
        assert np.array_equal(dones, ref.terminated | ref.truncated)
        assert np.allclose(obs, ref.obs)
        for i in range(4):
            info = infos[i]
            assert info["TimeLimit.truncated"] == bool(ref.truncated[i] and not ref.terminated[i])
            assert info["lap_completed"] == bool(ref.lap_completed[i])
            if dones[i]:
                seen_trunc |= bool(ref.truncated[i])
                seen_crash |= bool(ref.terminated[i])
                assert np.allclose(info["terminal_observation"], ref.final_obs[i])
                assert info["episode"]["r"] == np.float32(returns[i]) or np.isclose(info["episode"]["r"], returns[i])
                assert info["episode"]["l"] >= 1 and info["episode"]["t"] >= 0
                returns[i] = 0.0
            else:
                assert "terminal_observation" not in info and "episode" not in info
    assert seen_trunc and seen_crash


def test_ppo_learns_on_vec_env_smoke():
    """SB3's PPO accepts the VecEnv and fills its episode buffer from our "episode" infos."""
    vec = BatchSimVecEnv(load_track("oval"), EnvConfig(max_episode_s=2.0), n_envs=4, seed=0)
    model = PPO("MlpPolicy", vec, n_steps=64, batch_size=64, device="cpu", seed=0)
    model.learn(256)
    assert model.num_timesteps >= 256
    assert len(model.ep_info_buffer) > 0


def test_cli_knows_sb3():
    assert build_parser().parse_args(["train", "sb3", "--set", "track=oval"]).method == "sb3"
