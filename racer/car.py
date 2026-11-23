"""Batched kinematic-bicycle car model with a grip limit.

Why this exists: the simulator steps N cars at once every physics substep, so the
car is stored as parallel numpy arrays (one entry per car) and updated with no
Python loops over cars.

Conventions: world is y-up, meters, seconds, radians. ``heading`` is measured
counter-clockwise from +x. ``pos`` is the car's center (the rectangle's
midpoint). Simplification: the kinematic bicycle formally tracks the rear axle,
but here the center is used as the reference point. At the speeds and
wheelbase used here the difference is small and it keeps the geometry simple.

The grip limit is what makes braking for corners matter. A real car cannot turn
sharper than its tires allow: lateral acceleration is ``speed * yaw_rate =
speed**2 * curvature``. We cap curvature at ``max_lat_accel / speed**2``, so at
high speed the car understeers (it just refuses to turn that tightly) instead of
sliding. To get round a tight corner the agent must slow down first.
"""

from dataclasses import dataclass

import numpy as np

from racer.config import CarParams


@dataclass
class CarState:
    """State of N cars. Every field is an array with one entry per car."""

    pos: np.ndarray  # (N, 2) center position, m
    heading: np.ndarray  # (N,) radians CCW from +x
    speed: np.ndarray  # (N,) forward speed, m/s, never negative (no reverse)
    steer: np.ndarray  # (N,) current front-wheel angle, rad

    @classmethod
    def zeros(cls, n: int) -> "CarState":
        """N cars at the origin, facing +x, at rest, wheels straight."""
        return cls(np.zeros((n, 2)), np.zeros(n), np.zeros(n), np.zeros(n))

    def copy(self) -> "CarState":
        return CarState(self.pos.copy(), self.heading.copy(), self.speed.copy(), self.steer.copy())


def step_cars(state: CarState, actions: np.ndarray, p: CarParams, dt: float) -> CarState:
    """Advance every car by ``dt`` seconds and return a new state (input is untouched).

    ``actions`` is (N, 2): ``[steer, throttle]`` per car, each in [-1, 1].
    Negative throttle brakes; the car never reverses.
    """
    actions = np.clip(actions, -1.0, 1.0)
    steer_cmd, throttle = actions[:, 0], actions[:, 1]

    # The wheel angle moves toward the commanded angle, but only so fast.
    target = steer_cmd * p.max_steer
    max_delta = p.steer_rate * dt
    steer = state.steer + np.clip(target - state.steer, -max_delta, max_delta)

    # Longitudinal: engine (or brake) force minus quadratic drag, then clip.
    accel = np.where(throttle > 0.0, p.max_accel * throttle, p.max_brake * throttle)
    accel = accel - p.drag * state.speed**2
    speed = np.clip(state.speed + accel * dt, 0.0, p.max_speed)

    # Bicycle model: curvature from wheel angle, capped by the grip limit.
    curvature = np.tan(steer) / p.wheelbase
    grip_curvature = p.max_lat_accel / np.maximum(speed**2, 1e-6)
    curvature = np.clip(curvature, -grip_curvature, grip_curvature)
    heading = state.heading + speed * curvature * dt

    # Semi-implicit Euler: position uses the *updated* heading and speed. This is
    # slightly more stable than using the old values and costs nothing extra.
    direction = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    pos = state.pos + direction * (speed * dt)[:, None]

    return CarState(pos, heading, speed, steer)


def car_corners(state: CarState, p: CarParams) -> np.ndarray:
    """Rectangle corners, shape (N, 4, 2), ordered front-left, front-right, rear-right, rear-left.

    "Left" is +90 degrees from the heading (y-up, counter-clockwise).
    """
    hl, hw = p.length / 2.0, p.width / 2.0
    # Corner offsets in the car's own frame: x forward, y left.
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    c, s = np.cos(state.heading), np.sin(state.heading)
    # Rotate each local offset by the heading, then translate to the car's center.
    x = c[:, None] * local[None, :, 0] - s[:, None] * local[None, :, 1]
    y = s[:, None] * local[None, :, 0] + c[:, None] * local[None, :, 1]
    return state.pos[:, None, :] + np.stack([x, y], axis=-1)


def car_edges(state: CarState, p: CarParams) -> tuple[np.ndarray, np.ndarray]:
    """The 4 edges of each car as (starts, ends), each (N, 4, 2).

    Each edge runs from one corner to the next, so together they close the rectangle.
    """
    starts = car_corners(state, p)
    ends = np.roll(starts, -1, axis=1)
    return starts, ends
