"""Lesson 03 experiment: split one episode's return into its reward terms.

Drives one car on the oval with the scripted P-controller from tests/helpers.py (throttle 0.4,
coasting above 20 m/s) for one full episode with the default env config, and adds up each term
of the reward formula in racer/sim.py separately:

    reward = progress_scale * progress - time_cost + lap_bonus * lapped - crash_penalty * crashed

It prints the track length, how the episode ended, the number of decisions, each term's total,
and checks that the terms add up to the return the simulator paid out.

Run: uv run python docs/lessons/scripts/reward_breakdown.py   (a few seconds)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for tests.helpers

from racer.config import EnvConfig  # noqa: E402
from racer.sim import BatchSim  # noqa: E402
from racer.track import load_track  # noqa: E402
from tests.helpers import ScriptedOvalPolicy  # noqa: E402


def main() -> None:
    track = load_track("oval")
    cfg = EnvConfig()
    sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
    policy = ScriptedOvalPolicy(cfg, track)
    obs = sim.reset()
    paid, decisions, laps, crashed, end = 0.0, 0, 0, 0, ""
    while sim.alive[0]:
        res = sim.step(policy.act(obs))
        obs = res.obs
        paid += float(res.reward[0])
        decisions += 1
        laps += int(res.lap_completed[0])
        if res.terminated[0]:
            end = "terminated"
            # A crash also terminates; the wrong-way rule terminates without the penalty.
            crashed = int(res.reward[0] < -cfg.crash_penalty / 2)
        elif res.truncated[0]:
            end = "truncated"
    progress = float(sim.traveled[0])
    terms = {
        f"progress  {cfg.progress_scale} * {progress:.1f} m": cfg.progress_scale * progress,
        f"time cost -{cfg.time_cost} * {decisions} decisions": -cfg.time_cost * decisions,
        f"lap bonus {cfg.lap_bonus} * {laps} laps": cfg.lap_bonus * laps,
        f"crash     -{cfg.crash_penalty} * {crashed}": -cfg.crash_penalty * crashed,
    }
    print(f"oval length {track.length:.2f} m; episode {end} after {sim.episode_time[0]:.2f} s "
          f"({decisions} decisions of {cfg.action_repeat} physics steps)")
    for label, value in terms.items():
        print(f"  {label:40} {value:+9.2f}")
    print(f"  {'sum of terms':40} {sum(terms.values()):+9.2f}")
    print(f"  {'return paid by BatchSim':40} {paid:+9.2f}")


if __name__ == "__main__":
    main()
