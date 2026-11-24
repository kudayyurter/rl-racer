"""Lesson 08: how episodes end (crash, wrong way, time limit, finished), across a whole run.

Reward hacking shows up in *how* episodes end before it shows up in lap times: a time cost that
is too big makes ending the episode early worth more than driving on, and the cheapest ways to
end it are crashing or turning round (the wrong-way rule ends an episode with no penalty).
metrics.csv records laps and distance of each evaluation, not how it ended, so this script
replays checkpoints and classifies the endings.

For each run directory it replays every ``--every``-th checkpoint (deterministic, the episode
``evaluate`` drives) under the run's own env config, unless ``--env`` names another one, and
prints the count of each ending per quarter of the run, plus the median of the laps' times.
With ``--noisy N`` it also drives N sampled episodes (as in training) with the final checkpoint
of A2C, PPO, REINFORCE or SB3 runs and counts their endings.

Run: uv run python docs/lessons/scripts/endings.py runs/ppo-<ts>-s0 ... [--every 4] [--noisy 32]
     [--env configs/env.toml]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from actor_of import gaussian_actor
from racer.config import EnvConfig, from_dict, load_env_config
from racer.policy import load_policy
from racer.sim import BatchSim
from racer.track import load_track

ENDINGS = ("crash", "wrong way", "time limit", "finished")


def drive(policy_act, track, env: EnvConfig, n: int):
    """n cars, one episode each. Returns (endings list, lap times list, episode seconds)."""
    sim = BatchSim(track, env, n_cars=n, seed=0, auto_reset=False)
    obs = sim.reset()
    endings = [""] * n
    laps: list[float] = []
    while sim.alive.any():
        alive = sim.alive.copy()
        res = sim.step(policy_act(obs))
        obs = res.obs
        laps.extend(res.lap_time[res.lap_completed].tolist())
        for i in np.flatnonzero(alive & (res.terminated | res.truncated)):
            if res.terminated[i]:
                # A crash pays the penalty this decision; the wrong-way rule ends without one.
                endings[i] = "crash" if res.reward[i] < -env.crash_penalty / 2 else "wrong way"
            else:
                endings[i] = "finished" if sim.laps[i] >= env.max_laps else "time limit"
    return endings, laps, sim.episode_time.copy()


def counts(endings) -> str:
    c = Counter(endings)
    return " ".join(f"{c[e]:3d}" for e in ENDINGS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--every", type=int, default=4, help="replay every k-th checkpoint")
    parser.add_argument("--noisy", type=int, default=0, help="sampled episodes, final checkpoint")
    parser.add_argument("--env", help="env config TOML to replay under (default: the run's own)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print("endings per quarter of the run: crash / wrong way / time limit / finished, "
          "median lap s")
    for arg in args.runs:
        path = Path(arg)
        config = json.loads((path / "config.json").read_text())
        env = load_env_config(args.env) if args.env else from_dict(EnvConfig, config["env"])
        track = load_track(config["train"]["track"])
        ckpts = sorted((path / "checkpoints").glob("ckpt_*.pt"))[::args.every]
        quarters: list[list[str]] = [[], [], [], []]
        lap_times: list[list[float]] = [[], [], [], []]
        last = int(ckpts[-1].stem.split("_")[1]) if ckpts else 1
        for p in ckpts:
            q = min(3, 4 * (int(p.stem.split("_")[1]) - 1) // last)
            ends, laps, _ = drive(load_policy(p).act, track, env, 1)
            quarters[q] += ends
            lap_times[q] += laps
        parts = []
        for q in range(4):
            med = f"{np.median(lap_times[q]):5.2f}" if lap_times[q] else "  -  "
            parts.append(f"Q{q + 1} {counts(quarters[q])} {med}")
        where = f" under {args.env}" if args.env else ""
        print(f"{path.name} ({len(ckpts)} checkpoints{where})")
        print("  " + " | ".join(parts))
        if args.noisy and config["method"] != "evolution":
            actor = gaussian_actor(load_policy(path / "checkpoints" / "latest.pt"))
            torch.manual_seed(0)

            def sample(obs):
                with torch.no_grad():
                    return actor.dist(torch.as_tensor(obs, dtype=torch.float32)).sample().numpy()

            ends, laps, secs = drive(sample, track, env, args.noisy)
            med = f"{np.median(laps):.2f} s" if laps else "-"
            print(f"  final checkpoint, {args.noisy} sampled episodes: crash / wrong way / time "
                  f"limit / finished = {counts(ends)}; median lap {med}; "
                  f"median episode {np.median(secs):.1f} s")


if __name__ == "__main__":
    main()
