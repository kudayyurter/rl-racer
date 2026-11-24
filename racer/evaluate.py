"""Evaluation: run one deterministic episode and report what the lessons quote.

Why this exists: training curves use noisy, sampled behavior. To compare methods fairly (and to
produce the ghost laps the viewer replays) we drive one car with the policy's deterministic
action and record what happened: return, distance, laps, and the pose trace of the fastest lap.
"""

from dataclasses import dataclass

import numpy as np

from racer.config import EnvConfig
from racer.policy import Policy
from racer.sim import BatchSim
from racer.track import Track


@dataclass
class EvalResult:
    ret: float              # summed reward
    progress_m: float       # meters driven along the track
    laps: int
    best_lap_s: float       # NaN if no lap was completed
    lap_t: np.ndarray       # (T,) seconds since the best lap started
    lap_poses: np.ndarray   # (T, 3) x, y, heading during the best lap; empty if no lap


def evaluate(policy: Policy, track: Track, cfg: EnvConfig) -> EvalResult:
    """Drive one car from the start line until its episode ends, with no exploration noise."""
    sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
    obs = sim.reset()

    times = [float(sim.episode_time[0])]
    poses = [_pose(sim)]
    ret = 0.0
    lap_ends = []   # (index into times/poses, lap time) for each completed lap
    while sim.alive[0]:
        result = sim.step(policy.act(obs))
        obs = result.obs
        ret += float(result.reward[0])
        times.append(float(sim.episode_time[0]))
        poses.append(_pose(sim))
        if result.lap_completed[0]:
            lap_ends.append((len(times) - 1, float(result.lap_time[0])))

    best_lap_s, lap_t, lap_poses = float("nan"), np.zeros(0), np.zeros((0, 3))
    if lap_ends:
        times, poses = np.array(times), np.array(poses)
        # Lap k starts where lap k-1 ended (the spawn for the first lap), so slice between them.
        starts = [0] + [end for end, _ in lap_ends[:-1]]
        best = min(range(len(lap_ends)), key=lambda k: lap_ends[k][1])
        start, (end, best_lap_s) = starts[best], lap_ends[best]
        lap_t = times[start:end + 1] - times[start]
        lap_poses = poses[start:end + 1]

    return EvalResult(ret=ret, progress_m=float(sim.traveled[0]), laps=int(sim.laps[0]),
                      best_lap_s=best_lap_s, lap_t=lap_t, lap_poses=lap_poses)


def _pose(sim: BatchSim) -> list[float]:
    return [float(sim.state.pos[0, 0]), float(sim.state.pos[0, 1]), float(sim.state.heading[0])]
