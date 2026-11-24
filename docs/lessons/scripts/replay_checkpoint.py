"""Lesson 04: what did a saved network actually do? Replay checkpoints and describe the episode.

For each generation given, loads ``checkpoints/ckpt_XXXXX.pt`` of the run, drives one evaluation
episode with it (deterministic, from the start line, the run's own env config) and prints how
long the episode lasted, how far the car got, its top speed, how it ended and the return. Useful
to tell apart, say, a car that crashes at speed from one that crawls until the time limit, which
can have similar returns.

Run: uv run python docs/lessons/scripts/replay_checkpoint.py runs/evolution-<timestamp>-s0 5 10
"""

import json
import sys

from racer.config import EnvConfig, from_dict
from racer.policy import load_policy
from racer.sim import BatchSim
from racer.track import load_track


def main() -> None:
    run, generations = sys.argv[1], [int(g) for g in sys.argv[2:]]
    config = json.loads(open(f"{run}/config.json").read())
    env = from_dict(EnvConfig, config["env"])
    track = load_track(config["train"]["track"])
    for gen in generations:
        policy = load_policy(f"{run}/checkpoints/ckpt_{gen:05d}.pt")
        sim = BatchSim(track, env, n_cars=1, auto_reset=False)
        obs = sim.reset()
        ret, top_speed = 0.0, 0.0
        while sim.alive[0]:
            result = sim.step(policy.act(obs))
            obs = result.obs
            ret += float(result.reward[0])
            top_speed = max(top_speed, float(sim.state.speed[0]))
            ended = "crashed or wrong way" if result.terminated[0] else "time limit or max laps"
        print(f"generation {gen:4d}: {float(sim.episode_time[0]):5.1f} s, "
              f"{float(sim.traveled[0]):7.1f} m, {int(sim.laps[0])} laps, "
              f"top speed {top_speed:4.1f} m/s, {ended}, return {ret:.2f}")


if __name__ == "__main__":
    main()
