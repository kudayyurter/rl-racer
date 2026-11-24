"""Lesson 05: what does REINFORCE's very first batch look like, and which way does it push?

Rebuilds iteration 1 of ``racer train reinforce`` exactly as training does (same seed, same
network initialisation, same sim, same ``collect_batch``), then sorts the 32 episodes by how they
ended and prints, per kind: how many, how long, how far the car got, the undiscounted return, the
discounted return from the first step (G_0) and the mean reward-to-go over all its steps (the
weights the policy gradient multiplies each step's grad log-prob by).

It also prints two reference numbers: the discounted return of a car that sits still for the
full 90 s, and of one that crashes on its first decision.

Run: uv run python docs/lessons/scripts/reinforce_first_batch.py oval 0
"""

import sys

import numpy as np
import torch

from racer.algos.reinforce import collect_batch, discounted_returns
from racer.config import load_env_config
from racer.nets_torch import GaussianActor
from racer.sim import BatchSim, obs_dim
from racer.track import load_track

GAMMA, N_CARS, EPISODES = 0.99, 16, 32


def main() -> None:
    track_name, seed = sys.argv[1], int(sys.argv[2])
    env_cfg = load_env_config("configs/env.toml")
    torch.manual_seed(seed)  # as train_reinforce: the actor's init, then action sampling
    torch.set_num_threads(2)
    track = load_track(track_name)
    sim = BatchSim(track, env_cfg, n_cars=N_CARS, seed=seed, auto_reset=True)
    actor = GaussianActor(obs_dim(env_cfg), 2, (64, 64))

    # Track the end reason alongside collect_batch by wrapping sim.step.
    reasons: list[str] = []
    progress: list[float] = []
    real_step = sim.step

    def step(action):
        traveled_before = sim.traveled.copy()
        result = real_step(action)
        for i in np.flatnonzero(result.terminated | result.truncated):
            # traveled is zeroed by the auto-reset, so report it as of just before this step.
            # A crash is the only way to earn -crash_penalty in one step; the other
            # termination is driving the wrong way.
            if result.terminated[i]:
                crashed = result.reward[i] < -env_cfg.crash_penalty / 2
                reasons.append("crash" if crashed else "wrong way")
            else:
                reasons.append("time limit" if sim.laps[i] == 0 else "max laps")
            progress.append(float(traveled_before[i]))
        return result

    sim.step = step
    episodes, steps = collect_batch(sim, actor, EPISODES)
    reasons, progress = reasons[:len(episodes)], progress[:len(episodes)]
    print(f"{track_name}, seed {seed}: first batch = {len(episodes)} episodes, "
          f"{steps:,} car-decisions simulated (unfinished episodes dropped)")

    for kind in ("crash", "wrong way", "time limit", "max laps"):
        idx = [k for k, r in enumerate(reasons) if r == kind]
        if not idx:
            print(f"  {kind:10s}: none")
            continue
        lens = [len(episodes[k][2]) for k in idx]
        undisc = [episodes[k][2].sum() for k in idx]
        g = [discounted_returns(episodes[k][2], GAMMA) for k in idx]
        g0 = [x[0] for x in g]
        mean_g = np.concatenate(g).mean()
        tail = np.concatenate([x[-100:] for x in g]).mean()
        print(f"  {kind:10s}: {len(idx):2d} episodes, {sum(lens):,} of the batch's steps\n"
              f"      length {np.mean(lens):6.1f} (min {min(lens)}, max {max(lens)}), "
              f"progress {np.mean([progress[k] for k in idx]):5.1f} m, "
              f"return {np.mean(undisc):7.2f}, G_0 {np.mean(g0):6.2f}\n"
              f"      mean reward-to-go: over all its steps {mean_g:6.2f}, "
              f"over its last 100 steps {tail:6.2f}")

    all_g = np.concatenate([discounted_returns(r, GAMMA) for _, _, r in episodes])
    print(f"  reward-to-go over the whole batch: max {all_g.max():.3f}, min {all_g.min():.3f}, "
          f"{int((all_g < 0).sum()):,} of {all_g.size:,} steps negative")

    T = int(round(env_cfg.max_episode_s / (env_cfg.dt * env_cfg.action_repeat)))
    sit = discounted_returns(np.full(T, -env_cfg.time_cost), GAMMA)[0]
    crash_now = -env_cfg.time_cost - env_cfg.crash_penalty
    print(f"reference: sit still {T} decisions -> G_0 = {sit:.3f} (undiscounted "
          f"{-env_cfg.time_cost * T:.2f}); crash on the first decision -> G_0 = {crash_now:.2f}")


if __name__ == "__main__":
    main()
