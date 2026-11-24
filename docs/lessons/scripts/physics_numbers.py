"""Lesson 02 measurement: the car's basic numbers, measured by stepping racer.car.step_cars.

Every figure is measured by running the real physics step at dt = 1/60 s with the default
CarParams, not computed from a formula, so it includes the discrete time step:

* how long the wheel takes to go from straight to full lock (steer-rate limit)
* turning radius at full lock at 5 m/s (geometry only) and at 20 / 30 m/s (grip-limited)
* time and distance from rest to 50 m/s at full throttle (quadratic drag included)
* speed at constant throttle 0.4 after 5, 10 and 60 s on an endless straight
* distance needed to brake from 31 m/s to 24.5 m/s at full brake

Run: uv run python docs/lessons/scripts/physics_numbers.py   (a second)
"""

import numpy as np

from racer.car import CarState, step_cars
from racer.config import CarParams

P = CarParams()
DT = 1 / 60


def one_car(speed: float = 0.0, steer: float = 0.0) -> CarState:
    s = CarState.zeros(1)
    s.speed[:] = speed
    s.steer[:] = steer
    return s


def main() -> None:
    # Steer-rate limit: steps until the wheel angle reaches max_steer from 0.
    s, steps = one_car(), 0
    while s.steer[0] < P.max_steer - 1e-12:
        s = step_cars(s, np.array([[1.0, 0.0]]), P, DT)
        steps += 1
    print(f"straight to full lock: {steps} steps = {steps * DT:.3f} s")

    # Turning radius at full lock: radius = speed / yaw rate, over one step at constant speed.
    for v in (5.0, 20.0, 30.0):
        s = one_car(speed=v, steer=P.max_steer)
        throttle = P.drag * v**2 / P.max_accel      # just enough throttle to cancel drag
        new = step_cars(s, np.array([[1.0, throttle]]), P, DT)
        radius = new.speed[0] * DT / (new.heading[0] - s.heading[0])
        print(f"full-lock turning radius at {v:4.1f} m/s: {radius:6.1f} m")

    # Full throttle from rest to the speed cap.
    s, t, dist = one_car(), 0.0, 0.0
    while s.speed[0] < P.max_speed - 1e-9:
        new = step_cars(s, np.array([[0.0, 1.0]]), P, DT)
        dist += float(np.linalg.norm(new.pos[0] - s.pos[0]))
        s, t = new, t + DT
    print(f"full throttle 0 -> {P.max_speed:.0f} m/s: {t:.2f} s, {dist:.0f} m")

    # Constant throttle 0.4: speed after 10 s and after 60 s.
    s = one_car()
    for k in range(1, 60 * 60 + 1):
        s = step_cars(s, np.array([[0.0, 0.4]]), P, DT)
        if k in (5 * 60, 10 * 60, 60 * 60):
            print(f"throttle 0.4 after {k * DT:4.0f} s: {s.speed[0]:.1f} m/s")

    # Braking distance from 31 to 24.5 m/s.
    s, dist = one_car(speed=31.0), 0.0
    while s.speed[0] > 24.5:
        new = step_cars(s, np.array([[0.0, -1.0]]), P, DT)
        dist += float(np.linalg.norm(new.pos[0] - s.pos[0]))
        s = new
    print(f"full brake 31 -> 24.5 m/s: {dist:.1f} m")


if __name__ == "__main__":
    main()
