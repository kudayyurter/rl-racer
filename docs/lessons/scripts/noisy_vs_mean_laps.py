"""Lessons 05-06: did a policy learn to lap? Its noisy policy and its deterministic mean, per seed.

Training drives the *sampled* action (the Gaussian around the network's mean, clipped to
[-1, 1] by the sim); every evaluation in metrics.csv drives the *mean*. For each run directory
given, this script loads the final checkpoint (``checkpoints/latest.pt``) and reports:
  - noisy policy: 32 cars, one episode each from the start line, actions sampled exactly as in
    training (torch seed 0, so the numbers are reproducible): how many episodes completed 0, 1
    and 2 laps, how many ended by crash or wrong way, the median top speed, and where the
    episodes ended (median and quartiles of the distance driven);
  - mean policy: the deterministic evaluation episode (what ``evaluate`` drives): laps, where it
    ended, top speed;
  - for both, two like-for-like numbers on the opening straight, before any car has reached a
    turn: the speed when the car first passes 150 m, and the average applied throttle (after
    clipping) over the decisions before 215 m. Whole-episode averages are not comparable: an
    episode that crashes at 225 m is all acceleration, a full lap includes all the braking;
  - from metrics.csv: how many deterministic evaluations ever completed a lap, the first and the
    last that did, and the final one.

Run: uv run python docs/lessons/scripts/noisy_vs_mean_laps.py runs/reinforce-<timestamp>-s0 ...
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from racer.config import EnvConfig, from_dict
from racer.policy import load_policy
from racer.runlog import RunDir
from racer.sim import BatchSim
from racer.track import load_track

N_NOISY = 32
SPEED_AT_M = 150.0      # where on the opening straight to read the speed
STRAIGHT_END_M = 215.0  # average the throttle over decisions before this distance


def drive(track, env, actor, n: int, noisy: bool) -> dict:
    """n cars, one episode each; sampled or mean actions. The sim clips actions itself."""
    sim = BatchSim(track, env, n_cars=n, seed=0, auto_reset=False)
    obs = sim.reset()
    crashed = np.zeros(n, dtype=bool)
    top = np.zeros(n)
    speed_at = np.full(n, np.nan)
    straight_throttle: list[float] = []
    while sim.alive.any():
        alive = sim.alive.copy()
        with torch.no_grad():
            dist = actor.dist(torch.as_tensor(obs, dtype=torch.float32))
            action = (dist.sample() if noisy else dist.mean).numpy()
        on_straight = alive & (sim.traveled < STRAIGHT_END_M)
        straight_throttle.extend(np.clip(action[on_straight, 1], -1.0, 1.0))
        result = sim.step(action)
        crashed |= result.terminated
        top = np.maximum(top, np.where(alive, sim.state.speed, 0.0))
        passed = alive & np.isnan(speed_at) & (sim.traveled >= SPEED_AT_M)
        speed_at[passed] = sim.state.speed[passed]
        obs = result.obs
    return {"laps": sim.laps.copy(), "crashed": crashed, "top": top,
            "progress": sim.traveled.copy(), "speed_at": speed_at,
            "straight_throttle": float(np.mean(straight_throttle))}


def main() -> None:
    torch.set_num_threads(2)
    for arg in sys.argv[1:]:
        path = Path(arg)
        config = json.loads((path / "config.json").read_text())
        env = from_dict(EnvConfig, config["env"])
        track = load_track(config["train"]["track"])
        ckpt = torch.load(path / "checkpoints" / "latest.pt", weights_only=False)
        actor = load_policy(path / "checkpoints" / "latest.pt").actor
        torch.manual_seed(0)
        noisy = drive(track, env, actor, N_NOISY, noisy=True)
        mean = drive(track, env, actor, 1, noisy=False)
        laps = np.bincount(noisy["laps"], minlength=3)
        q1, q2, q3 = np.percentile(noisy["progress"], [25, 50, 75])

        rows = RunDir.open(path).read_metrics()
        evals = [r for r in rows if not math.isnan(r["eval_laps"])]
        lapped = [r for r in evals if r["eval_laps"] > 0]

        print(f"{path.name} ({config['train']['track']}, seed {config['seed']}, final checkpoint = "
              f"iteration {ckpt['iteration']}, {ckpt['env_steps']:,} steps)")
        print(f"  noisy policy, {N_NOISY} episodes: laps 0/1/2 = {laps[0]}/{laps[1]}/{laps[2]}, "
              f"crash or wrong way {int(noisy['crashed'].sum())}, "
              f"median top speed {np.median(noisy['top']):.1f} m/s")
        print(f"      distance driven: median {q2:.0f} m (quartiles {q1:.0f}-{q3:.0f} m)")
        print(f"      opening straight: speed at {SPEED_AT_M:.0f} m median "
              f"{np.nanmedian(noisy['speed_at']):.1f} m/s, applied throttle before "
              f"{STRAIGHT_END_M:.0f} m {noisy['straight_throttle']:+.2f}")
        ended = "crash or wrong way" if mean["crashed"][0] else "no crash"
        print(f"  mean policy: {int(mean['laps'][0])} laps, {ended} after "
              f"{mean['progress'][0]:.0f} m, top speed {mean['top'][0]:.1f} m/s")
        print(f"      opening straight: speed at {SPEED_AT_M:.0f} m {mean['speed_at'][0]:.1f} m/s, "
              f"applied throttle before {STRAIGHT_END_M:.0f} m {mean['straight_throttle']:+.2f}")
        if lapped:
            print(f"  deterministic evaluations that lapped: {len(lapped)} of {len(evals)}; "
                  f"first at iteration {int(lapped[0]['iteration'])} "
                  f"({int(lapped[0]['env_steps']):,} steps), last at iteration "
                  f"{int(lapped[-1]['iteration'])} ({int(lapped[-1]['env_steps']):,} steps)")
        else:
            print(f"  deterministic evaluations that lapped: 0 of {len(evals)} (never)")
        print(f"  final deterministic evaluation: {int(evals[-1]['eval_laps'])} laps, "
              f"{evals[-1]['eval_progress_m']:.1f} m")


if __name__ == "__main__":
    main()
