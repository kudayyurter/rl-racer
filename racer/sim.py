"""Batched racing simulator: N cars on one track, with sensors, reward, laps and resets.

Why this exists: every learner in the project (neuroevolution, REINFORCE, A2C, PPO) and the
Gymnasium wrapper drive cars through this one class, so the reward, lap and termination rules
written here are exactly what every method optimizes. All N cars are stored as parallel numpy
arrays and stepped together; there are no Python loops over cars in the step path.

One agent *decision* (one ``step`` call) runs ``action_repeat`` physics substeps with the same
action. Per substep: move the cars, project them onto the centerline to measure progress,
count laps, and check for wall crashes. Per decision: cast the sensor rays once and build the
observation.

Progress is measured relative to where the car spawned, never relative to s = 0 (on the oval,
s = 0 is not the start line). ``traveled`` is the signed distance driven along the track since
spawning; a lap is completed every time it passes another full track length. Driving back and
forth over the start line therefore cannot fake a lap.

Episode ends: *terminated* (a real failure, no future reward) on a crash or after driving
``wrong_way_m`` backwards from the furthest point reached; *truncated* (cut short, not failed)
at the time limit or after ``max_laps`` laps. Terminated takes precedence, so at most one of the
two is True. Each end is reported once, on the decision it happens.
"""

import math
from dataclasses import dataclass

import numpy as np

from racer.car import CarState, car_edges, step_cars
from racer.config import EnvConfig
from racer.geometry import any_crossing, ray_distances
from racer.track import Track

# Samples searched either side of a car's last index when projecting it onto the centerline.
# A car moves at most 50 m/s * 1/60 s ~ 0.83 m per substep, far less than 10 samples (~20 m).
PROJECT_HALF = 10
# Wall samples checked either side of a car for crashes. The car is 4.5 m long, so walls more
# than a few samples (~2 m each) away cannot touch it; 6 leaves plenty of margin.
CRASH_HALF = 6


def obs_dim(cfg: EnvConfig) -> int:
    """Observation size: one value per ray, plus speed, steer angle and heading error."""
    return cfg.n_rays + 3


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles into [-pi, pi). Car headings are unbounded (they keep counting turns)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class StepResult:
    """Everything one decision produced, one entry per car."""

    obs: np.ndarray            # (N, D) observation to act on next (after any auto-reset)
    reward: np.ndarray         # (N,)
    terminated: np.ndarray     # (N,) bool: crash or wrong way
    truncated: np.ndarray      # (N,) bool: time limit or max laps
    lap_completed: np.ndarray  # (N,) bool
    lap_time: np.ndarray       # (N,) seconds, NaN unless a lap finished this decision
    final_obs: np.ndarray      # (N, D) observation before any auto-reset; == obs if not done


class BatchSim:
    """N cars driving one track. See the module docstring for the rules.

    With ``auto_reset=False`` a car that finishes its episode freezes in place (reward 0,
    ``alive`` False) until ``reset()``; that suits "run one episode for a whole population".
    With ``auto_reset=True`` it respawns immediately, so every car is always driving; that
    suits rollout collection for policy-gradient methods.
    """

    def __init__(self, track: Track, cfg: EnvConfig, n_cars: int, seed: int = 0,
                 auto_reset: bool = False):
        self.cfg = cfg
        self.n_cars = n_cars
        self.auto_reset = auto_reset
        # One generator for the whole lifetime: reset() does not reseed, so successive
        # episodes get different random starts while the run as a whole stays reproducible.
        self.rng = np.random.default_rng(seed)
        self.track = track
        self._next_track: Track | None = None

        # Ray directions relative to the heading, evenly spread over +-ray_spread_deg/2.
        half_spread = math.radians(cfg.ray_spread_deg) / 2.0
        self._ray_angles = np.linspace(-half_spread, half_spread, cfg.n_rays)
        # Integer substep count for the time limit, so float sums can't miss it by 1e-15.
        self._max_ticks = math.ceil(cfg.max_episode_s / cfg.dt - 1e-9)

        n = n_cars
        self.state = CarState.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.laps = np.zeros(n, dtype=int)
        self.last_rays = np.full((n, cfg.n_rays), cfg.ray_range)
        self._idx = np.zeros(n, dtype=int)      # nearest centerline sample (projection hint)
        self._s = np.zeros(n)                   # last projected arc length, [0, length)
        self._traveled = np.zeros(n)            # signed progress since spawn, m
        self._max_traveled = np.zeros(n)        # furthest progress reached, m
        self._ticks = np.zeros(n, dtype=int)    # physics substeps since spawn
        self._lap_start = np.zeros(n, dtype=int)  # tick of the spawn or the previous lap
        self._obs = np.zeros((n, obs_dim(cfg)))

    # ---------------------------------------------------------------- read-only views

    @property
    def episode_time(self) -> np.ndarray:
        """(N,) seconds since each car spawned."""
        return self._ticks * self.cfg.dt

    @property
    def traveled(self) -> np.ndarray:
        """(N,) signed meters driven along the track since each car spawned (read-only copy)."""
        return self._traveled.copy()

    # ---------------------------------------------------------------- spawning

    def set_track(self, track: Track) -> None:
        """Switch to ``track`` at the next ``reset()``.

        Cars mid-episode keep driving the old track until then (one sim holds one track, so
        an auto-reset of a single car also stays on the current track). Call ``reset()`` to
        start everyone on the new one.
        """
        self._next_track = track

    def reset(self) -> np.ndarray:
        """Respawn every car (on the pending track, if one was set). Returns obs (N, D)."""
        if self._next_track is not None:
            self.track, self._next_track = self._next_track, None
        everyone = np.arange(self.n_cars)
        self._spawn(everyone)
        self._observe(everyone)
        return self._obs.copy()

    def place_car(self, i: int, s: float, lateral: float = 0.0, heading_offset: float = 0.0,
                  speed: float = 0.0) -> None:
        """Put car ``i`` at arc length ``s`` as a fresh spawn (time, laps and progress zeroed).

        ``lateral`` is meters to the left of the centerline, ``heading_offset`` radians
        counter-clockwise from the track direction. Used by tests and the viewer.
        """
        cars = np.array([i])
        self._spawn_at(cars, np.array([s], dtype=float), lateral, heading_offset, speed)
        self._observe(cars)

    def _spawn(self, cars: np.ndarray) -> None:
        """Respawn ``cars`` at the start line, or at random arc lengths if random_start."""
        if self.cfg.random_start:
            s = self.rng.uniform(0.0, self.track.length, size=len(cars))
        else:
            s = np.full(len(cars), self.track.cum_len[self.track.start_index])
        self._spawn_at(cars, s, 0.0, 0.0, 0.0)

    def _spawn_at(self, cars: np.ndarray, s: np.ndarray, lateral: float,
                  heading_offset: float, speed: float) -> None:
        """Place ``cars`` at arc lengths ``s`` and reset all their episode bookkeeping."""
        track = self.track
        pos, heading = track.pose_at(s)
        left = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
        self.state.pos[cars] = pos + lateral * left
        self.state.heading[cars] = heading + heading_offset
        self.state.speed[cars] = speed
        self.state.steer[cars] = 0.0

        # Project once so the progress baseline is the same s the step loop will measure.
        sample = np.searchsorted(track.cum_len, s % track.length, side="right") - 1
        idx, proj_s = track.project(self.state.pos[cars], sample, half=PROJECT_HALF)
        self._idx[cars] = idx
        self._s[cars] = proj_s
        self._traveled[cars] = 0.0
        self._max_traveled[cars] = 0.0
        self._ticks[cars] = 0
        self._lap_start[cars] = 0
        self.laps[cars] = 0
        self.alive[cars] = True

    # ---------------------------------------------------------------- stepping

    def step(self, actions: np.ndarray) -> StepResult:
        """Run one decision (``action_repeat`` substeps) for every car. actions: (N, 2)."""
        cfg, track = self.cfg, self.track
        n = self.n_cars
        actions = np.asarray(actions, dtype=float)

        active = self.alive.copy()      # frozen (done, non-auto-reset) cars sit this out
        crashed = np.zeros(n, dtype=bool)
        lapped = np.zeros(n, dtype=bool)
        lap_time = np.full(n, np.nan)
        progress = np.zeros(n)

        for _ in range(cfg.action_repeat):
            # A car that crashed in an earlier substep is done: it stops moving and earning.
            moving = active & ~crashed
            self.state = _select(moving, step_cars(self.state, actions, cfg.car, cfg.dt),
                                 self.state)
            self._ticks += moving

            idx, s = track.project(self.state.pos, self._idx, half=PROJECT_HALF)
            # Change in s, wrapped: crossing s = 0 forwards is +small, not -length.
            ds = (s - self._s + track.length / 2) % track.length - track.length / 2
            ds = np.where(moving, ds, 0.0)
            self._idx = np.where(moving, idx, self._idx)
            self._s = np.where(moving, s, self._s)
            self._traveled += ds
            self._max_traveled = np.maximum(self._max_traveled, self._traveled)
            progress += ds

            # Checked per substep so the lap time is accurate to one physics step.
            new_lap = moving & (self._traveled >= (self.laps + 1) * track.length)
            lap_time = np.where(new_lap, (self._ticks - self._lap_start) * cfg.dt, lap_time)
            self._lap_start = np.where(new_lap, self._ticks, self._lap_start)
            self.laps += new_lap
            lapped |= new_lap

            seg_a, seg_b = track.wall_window(self._idx, half=CRASH_HALF)
            edge_a, edge_b = car_edges(self.state, cfg.car)
            crashed |= moving & any_crossing(edge_a, edge_b, seg_a, seg_b)

        wrong_way = self._max_traveled - self._traveled > cfg.wrong_way_m
        terminated = active & (crashed | wrong_way)
        truncated = active & ~terminated & ((self._ticks >= self._max_ticks)
                                            | (self.laps >= cfg.max_laps))

        reward = (cfg.progress_scale * progress - cfg.time_cost
                  + cfg.lap_bonus * lapped - cfg.crash_penalty * crashed)
        reward = np.where(active, reward, 0.0)

        everyone = np.arange(n)
        self._observe(everyone)
        final_obs = self._obs.copy()
        done = terminated | truncated
        if self.auto_reset:
            respawn = np.flatnonzero(done)
            if len(respawn):
                self._spawn(respawn)
                self._observe(respawn)
        else:
            self.alive &= ~done

        return StepResult(obs=self._obs.copy(), reward=reward, terminated=terminated,
                          truncated=truncated, lap_completed=lapped, lap_time=lap_time,
                          final_obs=final_obs)

    # ---------------------------------------------------------------- sensing

    def _observe(self, cars: np.ndarray) -> None:
        """Cast rays and rebuild the observation rows for ``cars`` (updates last_rays, _obs)."""
        cfg, track = self.cfg, self.track
        rays = _cast_rays(self.state.pos[cars], self.state.heading[cars], self._idx[cars],
                          self._ray_angles, track, cfg.ray_range)
        self.last_rays[cars] = rays

        _, track_heading = track.pose_at(self._s[cars])
        heading_error = wrap_angle(self.state.heading[cars] - track_heading)
        self._obs[cars] = np.concatenate([
            rays / cfg.ray_range,
            (self.state.speed[cars] / cfg.car.max_speed)[:, None],
            (self.state.steer[cars] / cfg.car.max_steer)[:, None],
            (heading_error / np.pi)[:, None],
        ], axis=1)


def _cast_rays(pos: np.ndarray, heading: np.ndarray, idx: np.ndarray, ray_angles: np.ndarray,
               track: Track, ray_range: float) -> np.ndarray:
    """Distances (N, R) from each car's center along its ray fan to the nearest wall."""
    angles = heading[:, None] + ray_angles[None, :]                 # (N, R)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)      # (N, R, 2)
    # Enough wall samples either side to cover the full ray range, plus a little margin.
    half = math.ceil(ray_range / track.spacing) + 2
    seg_a, seg_b = track.wall_window(idx, half=half)
    return ray_distances(pos, dirs, seg_a, seg_b, ray_range)


def _select(mask: np.ndarray, new: CarState, old: CarState) -> CarState:
    """Per car: take ``new`` where mask is True, else keep ``old``."""
    return CarState(pos=np.where(mask[:, None], new.pos, old.pos),
                    heading=np.where(mask, new.heading, old.heading),
                    speed=np.where(mask, new.speed, old.speed),
                    steer=np.where(mask, new.steer, old.steer))
