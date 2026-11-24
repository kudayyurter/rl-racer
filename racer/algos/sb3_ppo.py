"""Stable-Baselines3 PPO: the answer key for our from-scratch PPO.

Why this exists: SB3 is a widely used, well-tested PPO. We train it on the same simulator
(one ``BatchSim`` of 16 cars, wrapped as an SB3 ``VecEnv`` below; ``RacerEnv`` stays the
single-car Gymnasium interface), with every hyperparameter set to the value our own PPO uses, and write
the same run-directory format (metrics columns, checkpoints, ghosts). ``watch`` and ``compare``
then treat it like any other method, and when its curve differs from ours the difference is
the implementation, not the settings. This is the only module that imports SB3.

How the two are matched (SB3 name <- our PPOConfig name / configs/ppo.toml key):

  SB3 argument                     ours                   notes
  -------------------------------  ---------------------  --------------------------------------
  n_steps                          n_steps                steps per env per rollout
  n_envs (BatchSimVecEnv)          n_cars                 16 cars in ONE BatchSim, like ours
  batch_size                       minibatch              samples per gradient step
  n_epochs                         epochs                 passes over each rollout
  gamma                            gamma
  gae_lambda                       lam
  clip_range                       clip
  ent_coef                         entropy_coef
  vf_coef                          value_coef
  max_grad_norm                    max_grad_norm
  learning_rate                    lr                     constant, no schedule
  normalize_advantage              normalize_adv          per minibatch in both
  net_arch pi=[64,64], vf=[64,64]  hidden                 separate actor and critic MLPs
  activation_fn=Tanh               (tanh in nets_torch)
  log_std_init=-0.5                LOG_STD_INIT           state-independent learned log std
  ortho_init=False                 (torch default init)   SB3 would use orthogonal init
  optimizer_kwargs eps=1e-8        (torch Adam default)   SB3 would use eps=1e-5

The last three rows are the settings SB3 does differently by default, so they are overridden.
The matched values live in configs/sb3.toml; tests/test_sb3.py asserts they equal ppo.toml.

Batched simulator: SB3 used to get 16 separate single-car BatchSims behind a DummyVecEnv, each
stepped from Python, which ran ~5x slower than our PPO for harness reasons alone. BatchSimVecEnv
steps one BatchSim of n_envs cars per call, exactly as our PPO does, so env_steps per wall-clock
second now compares the algorithms and not the plumbing.

Time limits: our env reports a time-limit ending as ``truncated`` (not ``terminated``). SB3's
vector env records that as ``TimeLimit.truncated`` and the PPO rollout bootstraps the missing
future with gamma * V(final observation), exactly what our PPO does in
``add_truncation_bootstrap``. That is why the comparison with our truncation handling is fair.
"""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from racer.config import EnvConfig
from racer.evaluate import evaluate
from racer.nets_torch import LOG_STD_INIT
from racer.policy import register_loader
from racer.runlog import RunDir
from racer.sim import BatchSim, obs_dim
from racer.track import Track, load_track

# SB3 reports no value loss or KL through our channel, so only the standard columns are logged.
EXTRA_COLUMNS: tuple[str, ...] = ()


@dataclass
class SB3Config:
    env_steps_budget: int           # stop after this many car-decisions
    eval_every_steps: int = 10_240  # evaluate + checkpoint roughly this often (on rollout ends)
    track: str = "twisty"
    n_envs: int = 16                # = our n_cars
    n_steps: int = 128              # per env
    batch_size: int = 256           # = our minibatch
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    normalize_advantage: bool = True
    net_arch: tuple[int, ...] = (64, 64)
    seed: int = 0


class SB3Policy:
    """A trained SB3 model as a Policy: the deterministic action, clipped to [-1, 1].

    The viewer's network panel reads the actor's layers: ``mlp_extractor.policy_net`` (linear +
    tanh pairs) followed by ``action_net`` (the linear layer that outputs the action mean).
    """

    def __init__(self, model: PPO):
        self.model = model
        policy = model.policy
        self._layers = list(policy.mlp_extractor.policy_net) + [policy.action_net]

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return np.clip(action, -1.0, 1.0).astype(float)

    def activations(self, obs: np.ndarray) -> list[np.ndarray]:
        """[input, each hidden layer after tanh, output mean] as numpy, like ``torch_activations``."""
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        acts = [x.numpy().astype(float)]
        with torch.no_grad():
            for layer in self._layers:
                x = layer(x)
                if isinstance(layer, torch.nn.Tanh) or layer is self._layers[-1]:
                    acts.append(x.numpy().astype(float))
        return acts

    def layer_weights(self) -> list[np.ndarray]:
        """Weight matrices shaped (in, out); nn.Linear stores (out, in), hence the transpose."""
        return [layer.weight.detach().numpy().T.astype(float)
                for layer in self._layers if isinstance(layer, torch.nn.Linear)]


def _load_policy(ckpt: dict) -> SB3Policy:
    """Rebuild the policy from the zip the checkpoint points at.

    ``weights`` is relative to the run directory; ``checkpoint_path`` (added by
    ``policy_from_checkpoint``) is ``<run>/checkpoints/latest.pt``, so the run is two levels up.
    """
    run_dir = Path(ckpt["checkpoint_path"]).parent.parent
    return SB3Policy(PPO.load(run_dir / ckpt["weights"], device="cpu"))


register_loader("sb3", _load_policy)


class BatchSimVecEnv(VecEnv):
    """One ``BatchSim`` of ``n_envs`` cars as an SB3 ``VecEnv``.

    Follows SB3's ``DummyVecEnv`` conventions so PPO treats it like any vector env:
    - ``dones = terminated | truncated``; the sim auto-resets, so the returned obs of a done
      car is already its respawn observation and the pre-reset one goes in
      ``info["terminal_observation"]``.
    - ``info["TimeLimit.truncated"]`` is True only for truncation (not a crash), which tells PPO
      to bootstrap from the value of the final observation.
    - ``info["episode"] = {"r", "l", "t"}`` on finished episodes: what SB3's ``Monitor`` wrapper
      would add, and what fills ``model.ep_info_buffer``.
    Observation and action spaces are identical to ``RacerEnv``'s.
    """

    def __init__(self, track: Track, env_cfg: EnvConfig, n_envs: int, seed: int = 0):
        super().__init__(n_envs,
                         spaces.Box(-1.0, 1.0, (obs_dim(env_cfg),), dtype=np.float32),
                         spaces.Box(-1.0, 1.0, (2,), dtype=np.float32))
        # auto_reset=True: a finished car respawns inside step(), so every car always drives.
        self.sim = BatchSim(track, env_cfg, n_cars=n_envs, seed=seed, auto_reset=True)
        self._actions = np.zeros((n_envs, 2))
        self._ep_return = np.zeros(n_envs)   # what Monitor tracks per env
        self._ep_len = np.zeros(n_envs, dtype=int)
        self._start = time.time()

    def reset(self):
        # The sim's RNG is seeded once at construction, so VecEnv.seed() has no effect here.
        self._ep_return[:] = 0.0
        self._ep_len[:] = 0
        return self.sim.reset().astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)

    def step_wait(self):
        # With auto_reset the sim zeroes a finished car's lap count, so count laps ourselves:
        # laps before the step plus the lap (if any) finished in it, as RacerEnv reports.
        laps_before = self.sim.laps.copy()
        res = self.sim.step(self._actions)
        laps = laps_before + res.lap_completed
        dones = res.terminated | res.truncated
        rewards = res.reward.astype(np.float32)
        self._ep_return += res.reward
        self._ep_len += 1
        infos: list[dict] = []
        for i in range(self.num_envs):
            info = {"lap_completed": bool(res.lap_completed[i]), "laps": int(laps[i]),
                    "TimeLimit.truncated": bool(res.truncated[i] and not res.terminated[i])}
            if res.lap_completed[i]:
                info["lap_time"] = float(res.lap_time[i])
            if dones[i]:
                info["terminal_observation"] = res.final_obs[i].astype(np.float32)
                info["episode"] = {"r": float(self._ep_return[i]), "l": int(self._ep_len[i]),
                                   "t": round(time.time() - self._start, 6)}
                self._ep_return[i] = 0.0
                self._ep_len[i] = 0
            infos.append(info)
        return res.obs.astype(np.float32), rewards, dones, infos

    # --- The rest of the VecEnv interface. PPO.learn only touches reset, step, seed and
    # get_attr("render_mode") (inside VecEnv.__init__); there are no per-env objects to expose.

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list:
        if attr_name == "render_mode":
            return [None] * self.num_envs
        raise AttributeError(f"BatchSimVecEnv has no per-env attribute {attr_name!r}")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        raise NotImplementedError("BatchSimVecEnv has no per-env objects to set attributes on")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        raise NotImplementedError("BatchSimVecEnv has no per-env objects to call methods on")

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * self.num_envs


def make_envs(cfg: SB3Config, env_cfg: EnvConfig) -> BatchSimVecEnv:
    """``n_envs`` cars in one BatchSim, seeded with ``cfg.seed``."""
    return BatchSimVecEnv(load_track(cfg.track), env_cfg, n_envs=cfg.n_envs, seed=cfg.seed)


CONFIG_CLASS = SB3Config  # names the CLI looks up for every method (see racer/cli.py)


def train_sb3(cfg: SB3Config, env_cfg: EnvConfig, run: RunDir) -> None:
    torch.set_num_threads(2)  # small nets: more threads do not pay off (see reinforce.py)
    track = load_track(cfg.track)
    envs = make_envs(cfg, env_cfg)
    model = PPO(
        "MlpPolicy", envs, seed=cfg.seed, device="cpu", verbose=0,
        n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
        learning_rate=cfg.learning_rate, normalize_advantage=cfg.normalize_advantage,
        policy_kwargs=dict(net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
                           activation_fn=torch.nn.Tanh, log_std_init=LOG_STD_INIT,
                           ortho_init=False, optimizer_kwargs=dict(eps=1e-8)))
    run.save_config({"method": "sb3", "seed": cfg.seed, "train": asdict(cfg),
                     "env": asdict(env_cfg)})
    rollout = cfg.n_steps * cfg.n_envs
    print(f"sb3 ppo on {track.name}: {cfg.n_envs} envs x {cfg.n_steps} steps per update, "
          f"{cfg.n_epochs} epochs of {cfg.batch_size}, actor and critic "
          f"{obs_dim(env_cfg)}-{'-'.join(map(str, cfg.net_arch))}")

    # SB3 has no "after the update" hook, so instead of a callback we call learn() in chunks of
    # whole rollouts and evaluate between chunks. reset_num_timesteps=False makes each call
    # continue the same environments and counters. Evaluation therefore always sees the policy
    # after its latest update, like our PPO's evaluations.
    chunk = max(1, math.ceil(cfg.eval_every_steps / rollout)) * rollout
    (run.path / "sb3_models").mkdir(exist_ok=True)
    start = time.time()
    while model.num_timesteps < cfg.env_steps_budget:
        remaining = cfg.env_steps_budget - model.num_timesteps
        model.learn(total_timesteps=min(chunk, remaining), reset_num_timesteps=False)
        _evaluate_and_save(model, cfg, env_cfg, track, run, start)


def _evaluate_and_save(model: PPO, cfg: SB3Config, env_cfg: EnvConfig, track, run: RunDir,
                       start: float) -> None:
    """Evaluate with our harness, then write the metrics row, checkpoint and ghost."""
    env_steps = model.num_timesteps
    iteration = env_steps // (cfg.n_steps * cfg.n_envs)  # completed rollouts, like PPO's
    # SB3 keeps the last 100 finished episodes (Monitor wrapper); a rolling window rather than
    # "episodes since the last log" as in our methods, but the same quantity: undiscounted return.
    returns = [ep["r"] for ep in model.ep_info_buffer]
    train_return = float(np.mean(returns)) if returns else np.nan
    res = evaluate(SB3Policy(model), track, env_cfg)

    zip_rel = Path("sb3_models") / f"model_{iteration:05d}.zip"
    model.save(run.path / zip_rel)  # SB3 keeps ".zip" when the name already ends in it
    run.save_checkpoint({"method": "sb3", "iteration": iteration, "env_steps": env_steps,
                         "obs_dim": obs_dim(env_cfg),
                         "arch": {"hidden": list(cfg.net_arch), "act_dim": 2},
                         "weights": str(zip_rel), "env_config": asdict(env_cfg)}, iteration)
    if res.laps > 0:
        run.save_ghost(iteration, res.lap_t, res.lap_poses, res.best_lap_s)
    run.log({"iteration": iteration, "env_steps": env_steps, "wall_time_s": time.time() - start,
             "train_return_mean": train_return, "eval_return": res.ret,
             "eval_progress_m": res.progress_m, "eval_laps": res.laps,
             "eval_best_lap_s": res.best_lap_s})
    line = (f"iter {iteration:4d}  steps {env_steps:>9,}  return {train_return:8.2f}  "
            f"| eval laps {res.laps}  progress {res.progress_m:6.1f} m")
    if res.laps > 0:
        line += f"  best lap {res.best_lap_s:.2f} s"
    print(line, flush=True)


train = train_sb3
