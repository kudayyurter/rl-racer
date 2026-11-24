"""Gymnasium wrapper: one car on one track, as a standard ``gymnasium.Env``.

Why this exists: Stable-Baselines3 is our "answer key" for the from-scratch PPO, and it only
speaks Gymnasium. This wrapper drives a ``BatchSim`` with a single car, so SB3 trains on exactly
the same physics, sensors, reward and termination rules as every method we write ourselves.
Nothing about the game is reimplemented here; it only adapts shapes and types.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from racer.config import EnvConfig, load_env_config
from racer.sim import BatchSim, obs_dim
from racer.track import Track, load_track

# Resolved from the package location so it works from any working directory.
DEFAULT_ENV_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "env.toml"


class RacerEnv(gym.Env):
    """Single-car racing environment. Observations and actions follow the BatchSim layout:
    action = (steer, throttle), each in [-1, 1]."""

    metadata = {"render_modes": []}

    def __init__(self, track: str | Track = "twisty", env_config: EnvConfig | None = None):
        super().__init__()
        self.track = load_track(track) if isinstance(track, str) else track
        self.cfg = env_config if env_config is not None else load_env_config(DEFAULT_ENV_CONFIG)
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim(self.cfg),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._sim = self._make_sim(seed=0)

    def _make_sim(self, seed: int) -> BatchSim:
        # auto_reset=False: Gymnasium callers call reset() themselves when an episode ends.
        return BatchSim(self.track, self.cfg, n_cars=1, seed=seed, auto_reset=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            # BatchSim's RNG is created at construction and never reseeded, so a new seed
            # means a new BatchSim (cheap). This makes reset(seed=s) reproducible.
            self._sim = self._make_sim(seed)
        obs = self._sim.reset()
        return obs[0].astype(np.float32), {}

    def step(self, action):
        # Check the shape before anything reshapes it: BatchSim would happily broadcast a
        # wrong-sized action, and reshape(1, 2) would accept a (2, 1) column or a (1, 2) batch.
        if np.shape(action) != (2,):
            raise ValueError(f"expected one (steer, throttle) action of shape (2,), "
                             f"got shape {np.shape(action)}")
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0).reshape(1, 2)
        res = self._sim.step(action)
        info = {"lap_completed": bool(res.lap_completed[0]), "laps": int(self._sim.laps[0])}
        if res.lap_completed[0]:
            # Only present on the step a lap finishes. A NaN placeholder would make check_env's
            # determinism test fail (NaN != NaN), and SB3 handles absent info keys fine.
            info["lap_time"] = float(res.lap_time[0])
        return (res.obs[0].astype(np.float32), float(res.reward[0]),
                bool(res.terminated[0]), bool(res.truncated[0]), info)
