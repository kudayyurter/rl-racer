"""Lesson 05: why does a noisy throttle drive slower than its mean, even when the mean is 0.35?

The car turns a throttle command c in [-1, 1] into an acceleration of ``max_accel * c`` for
c > 0 and ``max_brake * c`` for c < 0 (racer/car.py). With the defaults the brake is twice as
strong as the engine (16 against 8 m/s^2). So noise around a throttle mean does not average out:
a sample 0.6 below the mean brakes harder than a sample 0.6 above it accelerates.

For a throttle mean ``mu`` and noise std ``sigma`` this prints the expected acceleration:
  - of the mean itself (no noise),
  - with noise and clipping to [-1, 1], if the brake were as strong as the engine (clipping only),
  - with noise, clipping and the real, asymmetric brake,
and the speed each would reach after D meters from standstill, v = sqrt(2 a D) (ignoring drag,
which is small below 30 m/s). Expectations are Monte Carlo over 2,000,000 samples (seed 0).

Run: uv run python docs/lessons/scripts/noisy_throttle.py 0.35 0.6 150
"""

import sys

import numpy as np

from racer.config import load_env_config


def main() -> None:
    mu, sigma, dist = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    car = load_env_config("configs/env.toml").car
    c = np.clip(mu + sigma * np.random.default_rng(0).standard_normal(2_000_000), -1.0, 1.0)

    def accel(cmd, brake):
        return np.where(cmd > 0.0, car.max_accel * cmd, brake * cmd)

    cases = [("mean only, no noise", float(accel(np.array([mu]), car.max_brake)[0])),
             ("noise + clipping, brake as strong as engine", float(accel(c, car.max_accel).mean())),
             ("noise + clipping, real brake", float(accel(c, car.max_brake).mean()))]
    print(f"throttle mean {mu}, std {sigma}; max_accel {car.max_accel}, max_brake {car.max_brake} m/s^2")
    print(f"  samples below 0 (braking): {(c < 0).mean():.1%}, clipped at +1: {(c >= 1).mean():.1%}")
    for name, a in cases:
        print(f"  {name:45s} E[accel] {a:5.2f} m/s^2  ->  {np.sqrt(2 * a * dist):4.1f} m/s after {dist:.0f} m")


if __name__ == "__main__":
    main()
