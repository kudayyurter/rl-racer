"""A2C: REINFORCE plus a learned critic, short rollouts and bootstrapped n-step returns.

Read this file with ``diff racer/algos/reinforce.py racer/algos/a2c.py``; the diff is the lesson.
The loop is the same four steps, with these changes:

  1. COLLECT   Fixed-length rollouts: every car drives ``n_steps`` steps, then we update. Cars
               are NOT reset between rollouts; they carry on, so an episode may span several
               rollouts. (REINFORCE needed whole episodes; bootstrapping removes that need.)
  2. RETURNS   n-step returns: R[t] = r[t] + gamma * R[t+1], seeded at the rollout's end with
               the critic's estimate V(s_last) instead of 0. The critic stands in for "the
               rewards we have not seen yet". An episode boundary cuts the chain.
               Time-limit truncation is not a real ending: the car could have kept scoring, so
               at a truncated step we add gamma * V(final_obs) to its reward (REINFORCE's known
               bias, fixed). A crash really is the end, so terminated steps add nothing.
  3. LOSS      advantage A = R - V(s): "how much better than the critic expected". The actor
               loss is -mean(logp * A) (A is a constant to the actor), the critic regresses V(s)
               onto R, and an optional entropy bonus keeps exploration alive.
  4. UPDATE    One Adam step on actor and critic together, with the gradient norm clipped.

The baseline V(s) cuts the variance of the policy gradient without changing its direction.
The price: bias if the critic is wrong, and one more network to train.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from racer.algos.reinforce import TorchPolicy
from racer.config import EnvConfig
from racer.evaluate import evaluate
from racer.nets_torch import Critic, GaussianActor
from racer.policy import register_loader
from racer.runlog import RunDir
from racer.sim import BatchSim, obs_dim
from racer.track import load_track

# Extra metrics.csv columns: episode length, the policy's mean std, and the critic's health.
# explained_variance is 1 - Var(R - V) / Var(R): 1 = critic predicts returns perfectly,
# 0 = no better than a constant, negative = worse than a constant.
EXTRA_COLUMNS = ("episode_len_mean", "log_std_mean", "value_loss", "explained_variance")


@dataclass
class A2CConfig:
    env_steps_budget: int           # stop after this many car-decisions
    track: str = "twisty"
    n_cars: int = 16                # cars driving in parallel while collecting
    n_steps: int = 32               # steps per car per rollout; one update uses n_cars * n_steps
    gamma: float = 0.99             # discount: how much later rewards count
    lr: float = 7e-4                # Adam learning rate
    value_coef: float = 0.5         # weight of the critic's loss relative to the actor's
    entropy_coef: float = 0.0       # weight of the entropy bonus (0 = off)
    max_grad_norm: float = 0.5      # gradient clipping threshold
    hidden: tuple[int, ...] = (64, 64)
    eval_every: int = 10            # iterations between evaluations + checkpoints
    seed: int = 0


def _load_policy(ckpt: dict) -> TorchPolicy:
    actor = GaussianActor(ckpt["obs_dim"], ckpt["arch"]["act_dim"], tuple(ckpt["arch"]["hidden"]))
    actor.load_state_dict(ckpt["weights"])
    return TorchPolicy(actor)


register_loader("a2c", _load_policy)


# --------------------------------------------------------------------------- 1. COLLECT

def collect_rollout(sim: BatchSim, actor: GaussianActor, critic: Critic, obs: np.ndarray,
                    n_steps: int):
    """Drive every car ``n_steps`` steps from where the last rollout stopped.

    No reset: ``obs`` is each car's current observation (the sim auto-respawns finished cars).
    At a truncated step the car's real next state is ``final_obs`` (``obs`` is already the
    respawn), so we ask the critic what that state was worth and record it as
    ``final_values``; the caller folds it into the rewards (``add_truncation_bootstrap``).
    Returns (batch, next_obs):
      batch:    dict of arrays shaped (T, N, ...): ``obs``, ``actions`` (unclipped samples),
                ``rewards`` (raw, no bootstrap), ``terminated``, ``truncated``, and
                ``final_values`` = V(final_obs) at truncated steps (0 elsewhere); plus
                ``last_values`` (N,) = V of the observation after the last step
      next_obs: (N, D) where the next rollout continues
    """
    n = sim.n_cars
    buf = {k: [] for k in ("obs", "actions", "rewards", "terminated", "truncated", "final_values")}
    for _ in range(n_steps):
        with torch.no_grad():
            action = actor.dist(torch.as_tensor(obs, dtype=torch.float32)).sample().numpy()
        result = sim.step(action)  # the sim clips actions to [-1, 1] itself
        final_values = np.zeros(n)
        if result.truncated.any():
            with torch.no_grad():
                final_obs = torch.as_tensor(result.final_obs[result.truncated], dtype=torch.float32)
                final_values[result.truncated] = critic(final_obs).numpy()
        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["rewards"].append(result.reward)
        buf["terminated"].append(result.terminated)
        buf["truncated"].append(result.truncated)
        buf["final_values"].append(final_values)
        obs = result.obs
    batch = {k: np.array(v) for k, v in buf.items()}
    with torch.no_grad():
        batch["last_values"] = critic(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    return batch, obs


# --------------------------------------------------------------------------- 2. RETURNS

def add_truncation_bootstrap(rewards: np.ndarray, truncated: np.ndarray,
                             final_values: np.ndarray, gamma: float) -> np.ndarray:
    """Truncation is not a death: add gamma * V(final_obs) to the reward of truncated steps."""
    return rewards + gamma * final_values * truncated


def nstep_returns(rewards: np.ndarray, dones: np.ndarray, last_values: np.ndarray,
                  gamma: float) -> np.ndarray:
    """R[t] = r[t] + gamma * R[t+1], computed backwards from the rollout's end.

    rewards, dones: (T, N); last_values: (N,) is V of the state after the last step.
    ``dones[t]`` means the episode ended at step t (after r[t]), so R[t] does not bootstrap
    from t+1, which belongs to a different episode.
    """
    returns = np.zeros(rewards.shape)
    running = np.asarray(last_values, dtype=float)
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running
    return returns


# --------------------------------------------------------------------------- 3. LOSS

def a2c_loss(logp: torch.Tensor, values: torch.Tensor, returns: torch.Tensor, entropy: torch.Tensor,
             value_coef: float, entropy_coef: float):
    """Actor loss -mean(logp * A) + value_coef * MSE(V, R) - entropy_coef * mean(entropy).

    A = R - V is detached: the actor's gradient must not flow into the critic through it.
    Returns the total loss and the value loss (logged).
    """
    advantage = returns - values
    policy_loss = -(logp * advantage.detach()).mean()
    value_loss = (values - returns).pow(2).mean()
    return policy_loss + value_coef * value_loss - entropy_coef * entropy.mean(), value_loss


CONFIG_CLASS = A2CConfig  # names the CLI looks up for every method (see racer/cli.py)


def train_a2c(cfg: A2CConfig, env_cfg: EnvConfig, run: RunDir) -> None:
    torch.manual_seed(cfg.seed)  # network init and action sampling
    torch.set_num_threads(2)  # small nets: more threads do not pay off (see reinforce.py)
    track = load_track(cfg.track)
    sim = BatchSim(track, env_cfg, n_cars=cfg.n_cars, seed=cfg.seed, auto_reset=True)
    actor = GaussianActor(obs_dim(env_cfg), 2, cfg.hidden)
    critic = Critic(obs_dim(env_cfg), cfg.hidden)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    run.save_config({"method": "a2c", "seed": cfg.seed,
                     "train": asdict(cfg), "env": asdict(env_cfg)})
    print(f"a2c on {track.name}: {cfg.n_cars} cars x {cfg.n_steps} steps per update, "
          f"actor and critic {obs_dim(env_cfg)}-{'-'.join(map(str, cfg.hidden))}")

    obs = sim.reset()  # once: later rollouts continue the same cars
    car_return = np.zeros(cfg.n_cars)  # running return of each car's current episode
    car_len = np.zeros(cfg.n_cars)
    start, env_steps, iteration = time.time(), 0, 0
    while env_steps < cfg.env_steps_budget:
        iteration += 1

        # 1. COLLECT
        batch, obs = collect_rollout(sim, actor, critic, obs, cfg.n_steps)
        env_steps += cfg.n_steps * cfg.n_cars

        # 2. RETURNS (truncation bootstrap folded into the rewards, then the n-step recursion)
        dones = batch["terminated"] | batch["truncated"]
        rewards = add_truncation_bootstrap(batch["rewards"], batch["truncated"],
                                           batch["final_values"], cfg.gamma)
        returns = nstep_returns(rewards, dones, batch["last_values"], cfg.gamma)

        # 3. LOSS. Recompute log-probs and values with gradients from the stored observations.
        batch_obs = torch.as_tensor(batch["obs"].reshape(-1, batch["obs"].shape[-1]),
                                    dtype=torch.float32)
        batch_act = torch.as_tensor(batch["actions"].reshape(-1, 2), dtype=torch.float32)
        dist = actor.dist(batch_obs)
        logp = dist.log_prob(batch_act).sum(-1)  # sum over steer and throttle
        entropy = dist.entropy().sum(-1)
        values = critic(batch_obs)
        batch_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)
        loss, value_loss = a2c_loss(logp, values, batch_returns, entropy,
                                    cfg.value_coef, cfg.entropy_coef)

        # 4. UPDATE
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()

        # LOG / EVAL / CHECKPOINT. Episode returns count the real rewards (no bootstrap) of
        # episodes that finished in this rollout; they may have started in an earlier one.
        ep_returns, ep_lens = [], []
        for t in range(cfg.n_steps):
            car_return += batch["rewards"][t]
            car_len += 1
            for i in np.flatnonzero(dones[t]):
                ep_returns.append(car_return[i])
                ep_lens.append(car_len[i])
                car_return[i], car_len[i] = 0.0, 0.0
        train_return = float(np.mean(ep_returns)) if ep_returns else np.nan
        ep_len = float(np.mean(ep_lens)) if ep_lens else np.nan
        log_std = float(actor.log_std.detach().mean())
        var_ret = float(batch_returns.var())
        explained = (1.0 - float((batch_returns - values.detach()).var()) / var_ret
                     if var_ret > 0 else np.nan)
        row = {"iteration": iteration, "env_steps": env_steps,
               "wall_time_s": time.time() - start, "train_return_mean": train_return,
               "eval_return": np.nan, "eval_progress_m": np.nan, "eval_laps": np.nan,
               "eval_best_lap_s": np.nan, "episode_len_mean": ep_len, "log_std_mean": log_std,
               "value_loss": float(value_loss.detach()), "explained_variance": explained}
        line = (f"iter {iteration:4d}  steps {env_steps:>9,}  return {train_return:8.2f}  "
                f"ep len {ep_len:6.1f}  log_std {log_std:5.2f}  "
                f"v loss {float(value_loss.detach()):8.2f}  ev {explained:5.2f}")

        if iteration % cfg.eval_every == 0 or env_steps >= cfg.env_steps_budget:
            res = evaluate(TorchPolicy(actor), track, env_cfg)
            row.update(eval_return=res.ret, eval_progress_m=res.progress_m,
                       eval_laps=res.laps, eval_best_lap_s=res.best_lap_s)
            run.save_checkpoint({"method": "a2c", "iteration": iteration,
                                 "env_steps": env_steps, "obs_dim": obs_dim(env_cfg),
                                 "arch": {"hidden": list(cfg.hidden), "act_dim": 2},
                                 "weights": actor.state_dict(),
                                 "critic_weights": critic.state_dict(),
                                 "env_config": asdict(env_cfg)}, iteration)
            if res.laps > 0:
                run.save_ghost(iteration, res.lap_t, res.lap_poses, res.best_lap_s)
            line += f"  | eval laps {res.laps}  progress {res.progress_m:6.1f} m"
            if res.laps > 0:
                line += f"  best lap {res.best_lap_s:.2f} s"
        run.log(row)
        print(line, flush=True)


train = train_a2c
