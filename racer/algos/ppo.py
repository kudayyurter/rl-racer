"""PPO: A2C plus GAE, stored old log-probs and several clipped minibatch epochs per rollout.

Read this file with ``diff racer/algos/a2c.py racer/algos/ppo.py``; the diff is the lesson.
A2C throws each rollout away after one gradient step, because a second step on the same data
would push the policy far from the one that collected it, and the policy gradient is only
valid near that policy. PPO makes reuse safe. The loop is A2C's four steps, with these changes:

  1. COLLECT   Same rollout, but we also store each action's log-probability and the critic's
               value at collection time. The stored log-prob is the "old policy" PPO compares
               against.
  2. RETURNS   Generalized Advantage Estimation (GAE) replaces plain n-step returns.
               delta[t] = r[t] + gamma * V(s[t+1]) - V(s[t]) is a one-step advantage estimate;
               GAE sums those deltas with weights (gamma * lam)^k. lam = 0 gives the one-step
               estimate (low variance, more bias from the critic); lam = 1 gives the full
               n-step return minus V (A2C's advantage). lam ~ 0.95 sits in between. The
               critic's regression target is advantage + value.
  3. LOSS      ratio = pi_new(a|s) / pi_old(a|s) = exp(logp_new - logp_old). The clipped
               surrogate -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)) stops paying for
               moving the ratio beyond [1-eps, 1+eps] in the direction A favours, so the
               policy cannot run far from the old one. Value loss and entropy bonus as in A2C.
  4. UPDATE    Not one step but ``epochs`` passes over the rollout, each shuffled into
               minibatches, one Adam step (gradient norm clipped) per minibatch.

Everything else (rollout, truncation bootstrap, networks, logging) is unchanged from A2C.
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
# 0 = no better than a constant, negative = worse than a constant (measured with the values
# the critic gave at collection time, before this update's epochs).
# PPO adds two policy-movement gauges, averaged over all minibatch steps of the update:
#   approx_kl  how far the policy moved from the old one, mean((ratio - 1) - log ratio)
#              (a low-variance estimate of KL(old || new); ~0.01-0.02 is typical)
#   clip_frac  share of samples whose ratio left [1 - clip, 1 + clip] (clipping was active)
EXTRA_COLUMNS = ("episode_len_mean", "log_std_mean", "value_loss", "explained_variance",
                 "approx_kl", "clip_frac")


@dataclass
class PPOConfig:
    env_steps_budget: int           # stop after this many car-decisions
    track: str = "twisty"
    n_cars: int = 16                # cars driving in parallel while collecting
    n_steps: int = 128              # steps per car per rollout; one update uses n_cars * n_steps
    gamma: float = 0.99             # discount: how much later rewards count
    lam: float = 0.95               # GAE: 0 = one-step advantage, 1 = full n-step return - V
    clip: float = 0.2               # PPO clip range epsilon: ratio is kept near [1-eps, 1+eps]
    epochs: int = 10                # passes over each rollout's data
    minibatch: int = 256            # samples per gradient step
    lr: float = 3e-4                # Adam learning rate
    value_coef: float = 0.5         # weight of the critic's loss relative to the actor's
    entropy_coef: float = 0.0       # weight of the entropy bonus (0 = off)
    max_grad_norm: float = 0.5      # gradient clipping threshold
    normalize_adv: bool = True      # standardize advantages within each minibatch
    hidden: tuple[int, ...] = (64, 64)
    eval_every: int = 5             # iterations between evaluations + checkpoints
    seed: int = 0
    tracks: str = "fixed"           # "fixed": train on ``track``; "random" arrives with Task 15
    track_every: int = 1            # unused until random tracks exist (iterations per new track)


def _load_policy(ckpt: dict) -> TorchPolicy:
    actor = GaussianActor(ckpt["obs_dim"], ckpt["arch"]["act_dim"], tuple(ckpt["arch"]["hidden"]))
    actor.load_state_dict(ckpt["weights"])
    return TorchPolicy(actor)


register_loader("ppo", _load_policy)


# --------------------------------------------------------------------------- 1. COLLECT

def collect_rollout(sim: BatchSim, actor: GaussianActor, critic: Critic, obs: np.ndarray,
                    n_steps: int):
    """Drive every car ``n_steps`` steps from where the last rollout stopped.

    No reset: ``obs`` is each car's current observation (the sim auto-respawns finished cars).
    At a truncated step the car's real next state is ``final_obs`` (``obs`` is already the
    respawn), so we ask the critic what that state was worth and record it as
    ``final_values``; the caller folds it into the rewards (``add_truncation_bootstrap``).
    PPO also stores ``logp`` (the old policy's log-prob of each action) and ``values``
    (V of each observation), both fixed at collection time.
    Returns (batch, next_obs):
      batch:    dict of arrays shaped (T, N, ...): ``obs``, ``actions`` (unclipped samples),
                ``logp``, ``values``,
                ``rewards`` (raw, no bootstrap), ``terminated``, ``truncated``, and
                ``final_values`` = V(final_obs) at truncated steps (0 elsewhere); plus
                ``last_values`` (N,) = V of the observation after the last step
      next_obs: (N, D) where the next rollout continues
    """
    n = sim.n_cars
    buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "terminated",
                           "truncated", "final_values")}
    for _ in range(n_steps):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            dist = actor.dist(obs_t)
            action_t = dist.sample()
            logp = dist.log_prob(action_t).sum(-1).numpy()  # sum over steer and throttle
            value = critic(obs_t).numpy()
            action = action_t.numpy()
        result = sim.step(action)  # the sim clips actions to [-1, 1] itself
        final_values = np.zeros(n)
        if result.truncated.any():
            with torch.no_grad():
                final_obs = torch.as_tensor(result.final_obs[result.truncated], dtype=torch.float32)
                final_values[result.truncated] = critic(final_obs).numpy()
        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["logp"].append(logp)
        buf["values"].append(value)
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


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_values: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """Generalized Advantage Estimation, computed backwards from the rollout's end.

    delta[t] = r[t] + gamma * V[t+1] - V[t]      (one-step advantage)
    A[t]     = delta[t] + gamma * lam * A[t+1]   (decaying sum of later deltas)

    rewards, values, dones: (T, N); last_values: (N,) is V of the state after the last step.
    ``dones[t]`` means the episode ended at step t, so neither V[t+1] nor A[t+1] (which belong
    to a different episode) leak into step t. Returns advantages (T, N); the critic's
    regression targets are ``advantages + values``. With lam = 1 this is exactly
    ``nstep_returns - values``.
    """
    advantages = np.zeros(rewards.shape)
    running = np.zeros(rewards.shape[1])  # A[t+1]
    next_value = np.asarray(last_values, dtype=float)
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
        next_value = values[t]
    return advantages


# --------------------------------------------------------------------------- 3. LOSS

def clipped_surrogate(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                      clip: float) -> torch.Tensor:
    """PPO's policy loss: -mean(min(ratio * A, clip(ratio, 1-clip, 1+clip) * A)).

    ratio = pi_new / pi_old. The min keeps the pessimistic of the two terms: once the ratio has
    moved past the clip range in the direction the advantage likes, the clipped term wins and
    has zero gradient, so there is no more incentive to move further. (If the move was in the
    wrong direction, the unclipped term wins and still pushes back.) ``adv`` and ``logp_old``
    are constants.
    """
    ratio = (logp_new - logp_old).exp()
    return -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()


CONFIG_CLASS = PPOConfig  # names the CLI looks up for every method (see racer/cli.py)


def train_ppo(cfg: PPOConfig, env_cfg: EnvConfig, run: RunDir) -> None:
    if cfg.tracks != "fixed":
        raise NotImplementedError(f'tracks="{cfg.tracks}": only "fixed" exists so far')
    torch.manual_seed(cfg.seed)  # network init, action sampling and minibatch shuffling
    torch.set_num_threads(2)  # small nets: more threads do not pay off (see reinforce.py)
    track = load_track(cfg.track)
    sim = BatchSim(track, env_cfg, n_cars=cfg.n_cars, seed=cfg.seed, auto_reset=True)
    actor = GaussianActor(obs_dim(env_cfg), 2, cfg.hidden)
    critic = Critic(obs_dim(env_cfg), cfg.hidden)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    run.save_config({"method": "ppo", "seed": cfg.seed,
                     "train": asdict(cfg), "env": asdict(env_cfg)})
    print(f"ppo on {track.name}: {cfg.n_cars} cars x {cfg.n_steps} steps per update, "
          f"{cfg.epochs} epochs of {cfg.minibatch}, actor and critic {obs_dim(env_cfg)}-{'-'.join(map(str, cfg.hidden))}")

    obs = sim.reset()  # once: later rollouts continue the same cars
    car_return = np.zeros(cfg.n_cars)  # running return of each car's current episode
    car_len = np.zeros(cfg.n_cars)
    start, env_steps, iteration = time.time(), 0, 0
    while env_steps < cfg.env_steps_budget:
        iteration += 1

        # 1. COLLECT
        batch, obs = collect_rollout(sim, actor, critic, obs, cfg.n_steps)
        env_steps += cfg.n_steps * cfg.n_cars

        # 2. RETURNS (truncation bootstrap folded into the rewards, then GAE)
        dones = batch["terminated"] | batch["truncated"]
        rewards = add_truncation_bootstrap(batch["rewards"], batch["truncated"],
                                           batch["final_values"], cfg.gamma)
        advantages = compute_gae(rewards, batch["values"], dones, batch["last_values"],
                                 cfg.gamma, cfg.lam)
        returns = advantages + batch["values"]  # the critic's regression targets

        # Flatten (T, N) into one pool of samples. Everything the loss needs is fixed now.
        batch_obs = torch.as_tensor(batch["obs"].reshape(-1, batch["obs"].shape[-1]),
                                    dtype=torch.float32)
        batch_act = torch.as_tensor(batch["actions"].reshape(-1, 2), dtype=torch.float32)
        batch_logp_old = torch.as_tensor(batch["logp"].reshape(-1), dtype=torch.float32)
        batch_values = torch.as_tensor(batch["values"].reshape(-1), dtype=torch.float32)
        batch_adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32)
        batch_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)
        n_samples = len(batch_obs)

        # 3. LOSS + 4. UPDATE: several epochs, each a fresh shuffle cut into minibatches.
        value_losses, kls, clip_fracs = [], [], []
        for _ in range(cfg.epochs):
            order = torch.randperm(n_samples)
            for lo in range(0, n_samples, cfg.minibatch):
                idx = order[lo:lo + cfg.minibatch]
                dist = actor.dist(batch_obs[idx])
                logp = dist.log_prob(batch_act[idx]).sum(-1)  # sum over steer and throttle
                entropy = dist.entropy().sum(-1)
                values = critic(batch_obs[idx])
                adv = batch_adv[idx]
                if cfg.normalize_adv and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = clipped_surrogate(logp, batch_logp_old[idx], adv, cfg.clip)
                value_loss = (values - batch_returns[idx]).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    ratio = (logp - batch_logp_old[idx]).exp()
                    kls.append(float(((ratio - 1) - ratio.log()).mean()))
                    clip_fracs.append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
                value_losses.append(float(value_loss.detach()))

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
        value_loss = float(np.mean(value_losses))
        approx_kl, clip_frac = float(np.mean(kls)), float(np.mean(clip_fracs))
        var_ret = float(batch_returns.var())
        explained = (1.0 - float((batch_returns - batch_values).var()) / var_ret
                     if var_ret > 0 else np.nan)
        row = {"iteration": iteration, "env_steps": env_steps,
               "wall_time_s": time.time() - start, "train_return_mean": train_return,
               "eval_return": np.nan, "eval_progress_m": np.nan, "eval_laps": np.nan,
               "eval_best_lap_s": np.nan, "episode_len_mean": ep_len, "log_std_mean": log_std,
               "value_loss": value_loss, "explained_variance": explained,
               "approx_kl": approx_kl, "clip_frac": clip_frac}
        line = (f"iter {iteration:4d}  steps {env_steps:>9,}  return {train_return:8.2f}  "
                f"ep len {ep_len:6.1f}  log_std {log_std:5.2f}  "
                f"v loss {value_loss:8.2f}  ev {explained:5.2f}  "
                f"kl {approx_kl:.4f}  clip {clip_frac:.2f}")

        if iteration % cfg.eval_every == 0 or env_steps >= cfg.env_steps_budget:
            res = evaluate(TorchPolicy(actor), track, env_cfg)
            row.update(eval_return=res.ret, eval_progress_m=res.progress_m,
                       eval_laps=res.laps, eval_best_lap_s=res.best_lap_s)
            run.save_checkpoint({"method": "ppo", "iteration": iteration,
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


train = train_ppo
