"""Tests for the batched kinematic-bicycle car model (racer/car.py)."""

import numpy as np
import pytest

from racer.car import CarState, car_corners, car_edges, step_cars
from racer.config import CarParams

P = CarParams()
DT = 1 / 60


def run(state: CarState, action, steps: int, p: CarParams = P) -> CarState:
    """Step a state `steps` times with the same action for every car."""
    actions = np.tile(np.asarray(action, dtype=float), (len(state.speed), 1))
    for _ in range(steps):
        state = step_cars(state, actions, p, DT)
    return state


def test_zeros_and_copy():
    s = CarState.zeros(3)
    assert s.pos.shape == (3, 2) and s.heading.shape == (3,)
    c = s.copy()
    c.pos[0, 0] = 5.0
    assert s.pos[0, 0] == 0.0  # copy is deep


def test_step_is_pure():
    s = CarState.zeros(2)
    step_cars(s, np.array([[0.0, 1.0], [0.0, 1.0]]), P, DT)
    assert np.all(s.speed == 0.0) and np.all(s.pos == 0.0)


def test_full_throttle_one_second():
    s = run(CarState.zeros(1), [0.0, 1.0], 60)
    assert s.speed[0] == pytest.approx(P.max_accel * 1.0, rel=0.02)
    assert s.heading[0] == 0.0
    assert s.pos[0, 0] > 0.0
    assert s.pos[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_speed_capped():
    s = run(CarState.zeros(1), [0.0, 1.0], 20 * 60)
    assert s.speed[0] <= P.max_speed + 1e-6


def test_brake_never_reverses():
    s = CarState.zeros(1)
    s.speed[:] = 5.0
    s = run(s, [0.0, -1.0], 2 * 60)
    assert s.speed[0] == 0.0
    x = s.pos[0, 0]
    s = run(s, [0.0, -1.0], 30)
    assert s.pos[0, 0] == x  # stopped for good


def test_steer_rate_limited():
    s = run(CarState.zeros(1), [1.0, 0.0], 1)
    assert s.steer[0] == pytest.approx(P.steer_rate * DT)


def test_steer_reaches_but_never_exceeds_max():
    s = run(CarState.zeros(1), [1.0, 0.0], 60)
    assert s.steer[0] == pytest.approx(P.max_steer)


def test_grip_limit_caps_lateral_accel():
    s = CarState.zeros(1)
    s.speed[:] = 40.0
    s.steer[:] = P.max_steer  # full lock from the start; the grip limit must bind
    actions = np.array([[1.0, 0.3]])  # throttle keeps speed roughly constant
    for _ in range(2 * 60):
        prev = s.heading.copy()
        s = step_cars(s, actions, P, DT)
        # Heading wraps at +-pi, so unwrap the per-step change.
        dh = (s.heading - prev + np.pi) % (2 * np.pi) - np.pi
        lat = s.speed[0] * abs(dh[0] / DT)
        assert lat <= P.max_lat_accel * 1.01
    # And the limit really was binding (the car is turning, not going straight).
    assert lat > 0.9 * P.max_lat_accel


def test_low_speed_turns_tight():
    s = CarState.zeros(1)
    s.speed[:] = 5.0
    s.steer[:] = P.max_steer
    prev = s.heading.copy()
    s = step_cars(s, np.array([[1.0, 0.0]]), P, DT)
    radius = s.speed[0] * DT / (s.heading[0] - prev[0])
    assert radius == pytest.approx(P.wheelbase / np.tan(P.max_steer), rel=1e-3)


def test_corners_order_and_size():
    s = CarState.zeros(1)
    c = car_corners(s, P)
    assert c.shape == (1, 4, 2)
    hl, hw = P.length / 2, P.width / 2
    expected = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]  # FL, FR, RR, RL at heading 0
    assert np.allclose(c[0], expected)


def test_corners_rotate_and_translate():
    s = CarState.zeros(1)
    s.pos[:] = [10.0, 5.0]
    s.heading[:] = np.pi / 2  # facing +y, so "front" is up
    c = car_corners(s, P)
    assert np.allclose(c[0].mean(axis=0), [10.0, 5.0])
    assert c[0, 0, 1] > c[0, 3, 1]  # front-left is above rear-left
    assert c[0, 0, 0] < c[0, 1, 0]  # left (-x when facing +y) of right


def test_edges_close_the_rectangle():
    s = CarState.zeros(2)
    starts, ends = car_edges(s, P)
    assert np.allclose(ends, np.roll(starts, -1, axis=1))
    assert np.allclose(ends[:, -1], starts[:, 0])
