"""Lesson 05: does REINFORCE's "first 32 finished episodes" rule favour short episodes?

``collect_batch`` resets 16 cars and keeps the first 32 episodes that finish. A car whose first
episode is short gets to start a second one and may contribute both; a car stuck in a long
episode contributes at most one, and its unfinished second is dropped. So short episodes (crashes)
should be over-represented compared with the policy's true episode-length distribution.

This script measures it for one checkpoint. Unbiased reference: 32 cars, each drives exactly one
episode from the start line (``auto_reset=False``), every episode kept, however long. Batch rule:
``collect_batch`` with 16 cars and ``episodes`` episodes (32 by default), as training does.
Both sample actions from the
same policy; repeated over several random seeds and averaged.

Run: uv run python docs/lessons/scripts/batch_bias.py runs/reinforce-<timestamp>-s0 35 [episodes] [repeats]
     (checkpoint iteration; episodes per batch for collect_batch, default 32; repeats, default 10)
"""

import json
import sys

import numpy as np
import torch

from racer.algos.reinforce import collect_batch
from racer.config import EnvConfig, from_dict
from racer.policy import load_policy
from racer.sim import BatchSim
from racer.track import load_track


def one_episode_each(track, env, actor, n: int, seed: int) -> tuple[list[int], list[bool]]:
    """n cars, one whole episode each. Returns lengths and whether each crashed/went wrong way."""
    sim = BatchSim(track, env, n_cars=n, seed=seed, auto_reset=False)
    obs = sim.reset()
    lengths, terminated = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    while sim.alive.any():
        alive = sim.alive.copy()
        with torch.no_grad():
            action = actor.dist(torch.as_tensor(obs, dtype=torch.float32)).sample().numpy()
        result = sim.step(action)
        lengths += alive
        terminated |= result.terminated
        obs = result.obs
    return list(lengths), list(terminated)


def main() -> None:
    run, it = sys.argv[1], int(sys.argv[2])
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    config = json.loads(open(f"{run}/config.json").read())
    env = from_dict(EnvConfig, config["env"])
    track = load_track(config["train"]["track"])
    actor = load_policy(f"{run}/checkpoints/ckpt_{it:05d}.pt").actor
    torch.set_num_threads(2)

    ref_len, ref_term, batch_len, batch_term, batch_steps = [], [], [], [], []
    for seed in range(repeats):
        torch.manual_seed(seed)
        lengths, term = one_episode_each(track, env, actor, 32, seed)
        ref_len += lengths
        ref_term += term
        sim = BatchSim(track, env, n_cars=16, seed=seed, auto_reset=True)
        # Wrap step to record which episodes ended by termination, in the order they finish.
        ends: list[bool] = []
        real_step = sim.step

        def step(action, real_step=real_step, ends=ends):
            result = real_step(action)
            ends.extend(result.terminated[result.terminated | result.truncated])
            return result

        sim.step = step
        episodes, steps = collect_batch(sim, actor, n_episodes)
        batch_len += [len(r) for _, _, r in episodes]
        batch_term += ends[:len(episodes)]
        batch_steps.append(steps)

    print(f"{run} checkpoint {it}, {repeats} repeats; reference 32 episodes, "
          f"collect_batch {n_episodes} episodes")
    for name, lens, term in (("one episode per car (unbiased)", ref_len, ref_term),
                             (f"collect_batch, first {n_episodes} of 16 cars", batch_len, batch_term)):
        lens = np.array(lens)
        print(f"  {name:36s} mean length {lens.mean():6.1f}, median {np.median(lens):6.0f}, "
              f"crash or wrong way {np.mean(term):5.1%}, "
              f"episodes shorter than 300 decisions {(lens < 300).mean():5.1%}")
    kept = sum(batch_len)
    print(f"  collect_batch simulated {np.mean(batch_steps):,.0f} decisions per batch and kept "
          f"{kept / repeats:,.0f} ({kept / sum(batch_steps):.0%}); the rest were dropped "
          f"unfinished episodes")


if __name__ == "__main__":
    main()
