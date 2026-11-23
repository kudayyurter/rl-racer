"""Tests for the Gymnasium wrapper (racer/gym_env.py): API conformance and parity with BatchSim."""

import dataclasses

import numpy as np
from gymnasium.utils.env_checker import check_env

from racer.config import EnvConfig
from racer.gym_env import RacerEnv
from racer.sim import BatchSim, obs_dim
from racer.track import load_track


def test_passes_env_checker():
    # No rendering exists (the viewer is separate), so there are no render modes to check.
    check_env(RacerEnv("oval"), skip_render_check=True)


def test_spaces_match_simulator():
    env = RacerEnv("oval")
    assert env.observation_space.shape == (obs_dim(EnvConfig()),)
    assert env.action_space.shape == (2,)


def test_matches_batchsim():
    cfg = EnvConfig()
    sim = BatchSim(load_track("oval"), cfg, 1, seed=3)
    sim_obs = sim.reset()
    env = RacerEnv("oval", cfg)
    obs, _ = env.reset(seed=3)
    np.testing.assert_allclose(obs, sim_obs[0], atol=1e-6)

    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.uniform(-1, 1, size=2)
        res = sim.step(action[None, :])
        obs, reward, terminated, truncated, _ = env.step(action)
        assert reward == res.reward[0]
        assert terminated == res.terminated[0] and truncated == res.truncated[0]
        np.testing.assert_allclose(obs, res.obs[0], atol=1e-6)
        if terminated or truncated:
            break


def test_reset_seed_is_reproducible():
    cfg = dataclasses.replace(EnvConfig(), random_start=True)
    env = RacerEnv("oval", cfg)
    first, _ = env.reset(seed=11)
    env.reset(seed=12)
    again, _ = env.reset(seed=11)
    np.testing.assert_array_equal(first, again)


def test_out_of_range_actions_are_clipped():
    a, b = RacerEnv("oval"), RacerEnv("oval")
    a.reset(seed=0), b.reset(seed=0)
    obs_a = a.step(np.array([5.0, -9.0], dtype=np.float32))[0]
    obs_b = b.step(np.array([1.0, -1.0], dtype=np.float32))[0]
    np.testing.assert_array_equal(obs_a, obs_b)


def test_info_has_plain_python_values():
    env = RacerEnv("oval")
    env.reset(seed=0)
    info = env.step(env.action_space.sample())[4]
    assert type(info["lap_completed"]) is bool
    assert type(info["laps"]) is int
    assert "lap_time" not in info  # only reported on the step a lap finishes


def test_lap_time_reported_when_lap_finishes():
    env = RacerEnv("oval")
    env.reset(seed=0)
    # Pretend almost a full lap is already driven, so the lap finishes within a few steps.
    env._sim._traveled[0] = env.track.length - 1.0
    env._sim.state.speed[0] = 20.0
    lap_info = None
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        if info["lap_completed"] or terminated or truncated:
            lap_info = info
            break
    assert lap_info is not None and type(lap_info["lap_time"]) is float


def test_crash_terminates_episode():
    env = RacerEnv("oval")
    env.reset(seed=0)
    for _ in range(2000):
        _, _, terminated, truncated, _ = env.step(np.array([1.0, 1.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated
