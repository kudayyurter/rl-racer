"""REINFORCE: the simplest policy gradient. Learn by making good episodes more likely.

The algorithm, one *batch* at a time:

  1. COLLECT   n_cars cars drive in parallel with the current stochastic policy (sample actions
               around the network's mean). Each car's steps are kept as separate episodes. Stop
               once ``episodes_per_batch`` episodes have finished; unfinished ones are dropped.
  2. RETURNS   For every step, G = the discounted sum of rewards from that step to the episode's
               end ("reward-to-go"). It says how well things went *after* that decision.
  3. LOSS      loss = -mean(logp(action) * G). Minimising it raises the probability of actions
               followed by high G and lowers it for low G. This is the policy gradient.
  4. UPDATE    One Adam step on the actor. Then throw the batch away: the data came from the old
               policy, so it is only valid for one step (on-policy learning).

Deliberately plain: no baseline, no reward normalisation, no entropy bonus, no gradient
clipping. Expect noisy, slow learning; that noise is what A2C (a critic baseline) and PPO fix.
One known bias: when an episode ends by time limit or max laps (truncation) we still treat the
end as "nothing more to come", exactly like a crash. A2C fixes this by bootstrapping.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from racer.config import EnvConfig
from racer.evaluate import evaluate
from racer.nets_torch import GaussianActor, torch_activations, torch_layer_weights
from racer.policy import register_loader
from racer.runlog import RunDir
from racer.sim import BatchSim, obs_dim
from racer.track import load_track

# Extra metrics.csv columns: how long the batch's episodes lasted, and the policy's mean std.
EXTRA_COLUMNS = ("episode_len_mean", "log_std_mean")


@dataclass
class ReinforceConfig:
    env_steps_budget: int           # stop after this many car-decisions
    track: str = "twisty"
    n_cars: int = 16                # cars driving in parallel while collecting
    episodes_per_batch: int = 32    # finished episodes that make one gradient step
    gamma: float = 0.99             # discount: how much later rewards count
    lr: float = 3e-4                # Adam learning rate
    hidden: tuple[int, ...] = (64, 64)
    eval_every: int = 5             # iterations between evaluations + checkpoints
    seed: int = 0


class TorchPolicy:
    """A trained actor as a Policy: the deterministic mean action, clipped to [-1, 1]."""

    def __init__(self, actor: GaussianActor):
        self.actor = actor

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            mean = self.actor.mean(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
        return mean.clamp(-1.0, 1.0).numpy().astype(float)

    def activations(self, obs: np.ndarray) -> list[np.ndarray]:
        return torch_activations(self.actor, obs)

    def layer_weights(self) -> list[np.ndarray]:
        return torch_layer_weights(self.actor)


def _load_policy(ckpt: dict) -> TorchPolicy:
    actor = GaussianActor(ckpt["obs_dim"], ckpt["arch"]["act_dim"], tuple(ckpt["arch"]["hidden"]))
    actor.load_state_dict(ckpt["weights"])
    return TorchPolicy(actor)


register_loader("reinforce", _load_policy)


# --------------------------------------------------------------------------- 1. COLLECT

def collect_batch(sim: BatchSim, actor: GaussianActor, obs: np.ndarray, n_episodes: int):
    """Drive all cars until ``n_episodes`` episodes have finished.

    Each car keeps its own growing buffer. When a car's step reports terminated or truncated,
    that step's reward is the episode's last, so we close the episode and start a new buffer
    (the sim has already respawned the car). Returns:
      episodes: list of (obs (T, D), actions (T, 2), rewards (T,)) for finished episodes
      obs:      the observations to continue from next batch
      steps:    car-decisions simulated, including those in dropped unfinished buffers
    """
    n = sim.n_cars
    buf_obs: list[list[np.ndarray]] = [[] for _ in range(n)]
    buf_act: list[list[np.ndarray]] = [[] for _ in range(n)]
    buf_rew: list[list[float]] = [[] for _ in range(n)]
    episodes, steps = [], 0
    while len(episodes) < n_episodes:
        with torch.no_grad():
            action = actor.dist(torch.as_tensor(obs, dtype=torch.float32)).sample().numpy()
        result = sim.step(action)  # the sim clips actions to [-1, 1] itself
        steps += n
        for i in range(n):
            buf_obs[i].append(obs[i])
            buf_act[i].append(action[i])
            buf_rew[i].append(result.reward[i])
            if result.terminated[i] or result.truncated[i]:
                episodes.append((np.array(buf_obs[i]), np.array(buf_act[i]),
                                 np.array(buf_rew[i])))
                buf_obs[i], buf_act[i], buf_rew[i] = [], [], []
        obs = result.obs
    return episodes, obs, steps


# --------------------------------------------------------------------------- 2. RETURNS

def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Reward-to-go: G[t] = r[t] + gamma * G[t+1], computed backwards from the episode's end."""
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


# --------------------------------------------------------------------------- 3. LOSS

def reinforce_loss(logp: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """The policy-gradient loss: -mean(logp * G). Its gradient is the REINFORCE estimate."""
    return -(logp * returns).mean()


CONFIG_CLASS = ReinforceConfig  # names the CLI looks up for every method (see racer/cli.py)


def train_reinforce(cfg: ReinforceConfig, env_cfg: EnvConfig, run: RunDir) -> None:
    torch.manual_seed(cfg.seed)  # network init and action sampling
    # Measured on this machine: acting with 16 rows costs the same at 1, 2 or 4 threads, and the
    # big per-batch gradient step is ~40% faster at 2 than at 1 but no faster at 4. These nets
    # are too small for more threads to pay off, so use 2. (Set here, not at import, so merely
    # loading a checkpoint in the viewer does not change torch's global settings.)
    torch.set_num_threads(2)
    track = load_track(cfg.track)
    sim = BatchSim(track, env_cfg, n_cars=cfg.n_cars, seed=cfg.seed, auto_reset=True)
    actor = GaussianActor(obs_dim(env_cfg), 2, cfg.hidden)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    run.save_config({"method": "reinforce", "seed": cfg.seed,
                     "train": asdict(cfg), "env": asdict(env_cfg)})
    print(f"reinforce on {track.name}: {cfg.n_cars} cars, {cfg.episodes_per_batch} episodes per "
          f"batch, actor {obs_dim(env_cfg)}-{'-'.join(map(str, cfg.hidden))}-2")

    start, env_steps, iteration = time.time(), 0, 0
    obs = sim.reset()
    while env_steps < cfg.env_steps_budget:
        iteration += 1

        # 1. COLLECT
        episodes, obs, steps = collect_batch(sim, actor, obs, cfg.episodes_per_batch)
        env_steps += steps

        # 2. RETURNS (per episode, so reward-to-go never leaks across an episode boundary)
        returns = np.concatenate([discounted_returns(r, cfg.gamma) for _, _, r in episodes])

        # 3. LOSS. Recompute log-probs with gradients from the stored observations and actions;
        # the policy hasn't changed since collection, so this equals the sampling-time logp.
        batch_obs = torch.as_tensor(np.concatenate([o for o, _, _ in episodes]), dtype=torch.float32)
        batch_act = torch.as_tensor(np.concatenate([a for _, a, _ in episodes]), dtype=torch.float32)
        logp = actor.dist(batch_obs).log_prob(batch_act).sum(-1)  # sum over steer and throttle
        loss = reinforce_loss(logp, torch.as_tensor(returns, dtype=torch.float32))

        # 4. UPDATE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LOG / EVAL / CHECKPOINT
        ep_returns = [float(r.sum()) for _, _, r in episodes]
        ep_len = float(np.mean([len(r) for _, _, r in episodes]))
        log_std = float(actor.log_std.detach().mean())
        row = {"iteration": iteration, "env_steps": env_steps,
               "wall_time_s": time.time() - start, "train_return_mean": float(np.mean(ep_returns)),
               "eval_return": np.nan, "eval_progress_m": np.nan, "eval_laps": np.nan,
               "eval_best_lap_s": np.nan, "episode_len_mean": ep_len, "log_std_mean": log_std}
        line = (f"iter {iteration:4d}  steps {env_steps:>9,}  return {np.mean(ep_returns):8.2f}  "
                f"ep len {ep_len:6.1f}  log_std {log_std:5.2f}")

        if iteration % cfg.eval_every == 0 or env_steps >= cfg.env_steps_budget:
            res = evaluate(TorchPolicy(actor), track, env_cfg)
            row.update(eval_return=res.ret, eval_progress_m=res.progress_m,
                       eval_laps=res.laps, eval_best_lap_s=res.best_lap_s)
            run.save_checkpoint({"method": "reinforce", "iteration": iteration,
                                 "env_steps": env_steps, "obs_dim": obs_dim(env_cfg),
                                 "arch": {"hidden": list(cfg.hidden), "act_dim": 2},
                                 "weights": actor.state_dict(),
                                 "env_config": asdict(env_cfg)}, iteration)
            if res.laps > 0:
                run.save_ghost(iteration, res.lap_t, res.lap_poses, res.best_lap_s)
            line += f"  | eval laps {res.laps}  progress {res.progress_m:6.1f} m"
            if res.laps > 0:
                line += f"  best lap {res.best_lap_s:.2f} s"
        run.log(row)
        print(line, flush=True)


train = train_reinforce
