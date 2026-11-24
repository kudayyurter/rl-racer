"""Lesson 08: what the time cost pays for, under three environment configs.

Replays trained checkpoints (deterministic, one car, from the start line) under:

  default       configs/env.toml                 time_cost 0.01, max_laps 2, 90 s limit
  old fastlap   env.toml + time_cost 0.05 + max_laps 3 (the first env_fastlap.toml, 90 s limit)
  fastlap       configs/env_fastlap.toml         time_cost 0.05, max_laps 2, 120 s limit

and prints, per episode: how it ended, its length, the lap times, and the return split into
the four reward terms of racer/sim.py. The point to look for: under "old fastlap" every
driver that survives runs exactly 90 s (three twisty laps never fit), so the time cost is the
same 67.5 for all of them and cannot rank them; under "fastlap" the episode ends at the line,
so the time cost is 0.75 x episode length and a faster driver pays less.

Arguments are run directories, optionally RUN@ITERATION for checkpoints/ckpt_<iteration>.pt
(default checkpoints/latest.pt).

Run: uv run python docs/lessons/scripts/time_cost_returns.py runs/evolution-<ts>-s0 runs/ppo-<ts>-s0@110
"""

import sys
from pathlib import Path

import torch

from racer.config import load_env_config
from racer.policy import policy_from_checkpoint
from racer.sim import BatchSim
from racer.track import load_track

CONFIGS = {
    "default": load_env_config("configs/env.toml"),
    "old fastlap": load_env_config("configs/env.toml", {"time_cost": 0.05, "max_laps": 3}),
    "fastlap": load_env_config("configs/env_fastlap.toml"),
}


def checkpoint_path(arg: str) -> Path:
    run, _, iteration = arg.partition("@")
    name = f"ckpt_{int(iteration):05d}.pt" if iteration else "latest.pt"
    return Path(run) / "checkpoints" / name


def episode(policy, track, cfg) -> dict:
    """One deterministic episode; returns its ending, length, lap times and reward terms."""
    sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
    obs = sim.reset()
    paid, decisions, laps, crashed, end = 0.0, 0, [], 0, ""
    while sim.alive[0]:
        res = sim.step(policy.act(obs))
        obs = res.obs
        paid += float(res.reward[0])
        decisions += 1
        if res.lap_completed[0]:
            laps.append(float(res.lap_time[0]))
        if res.terminated[0]:
            # A crash pays the penalty; the wrong-way rule terminates without one.
            crashed = int(res.reward[0] < -cfg.crash_penalty / 2)
            end = "crash" if crashed else "wrong way"
        elif res.truncated[0]:
            end = f"{cfg.max_laps} laps" if len(laps) >= cfg.max_laps else "time limit"
    terms = {"progress": cfg.progress_scale * float(sim.traveled[0]),
             "time": -cfg.time_cost * decisions,
             "laps": cfg.lap_bonus * len(laps),
             "crash": -cfg.crash_penalty * crashed}
    assert abs(sum(terms.values()) - paid) < 1e-6, "reward terms do not add up"
    return {"end": end, "seconds": float(sim.episode_time[0]), "laps": laps,
            "meters": float(sim.traveled[0]), "terms": terms, "ret": paid}


def main() -> None:
    track = load_track("twisty")
    print(f"twisty: {track.length:.2f} m per lap")
    for arg in sys.argv[1:]:
        path = checkpoint_path(arg)
        ckpt = torch.load(path, weights_only=False)
        policy = policy_from_checkpoint(ckpt, path)
        print(f"\n{arg}  ({ckpt['method']}, iteration {ckpt['iteration']})")
        for name, cfg in CONFIGS.items():
            e = episode(policy, track, cfg)
            laps = ", ".join(f"{t:.2f}" for t in e["laps"]) or "-"
            t = e["terms"]
            print(f"  {name:11}  {e['end']:10}  {e['seconds']:6.2f} s  {e['meters']:7.1f} m  "
                  f"laps [{laps}]  return {e['ret']:7.2f} = progress {t['progress']:6.1f} "
                  f"time {t['time']:6.2f} laps {t['laps']:4.0f} crash {t['crash']:4.0f}")


if __name__ == "__main__":
    main()
