"""Typed, strictly validated configuration for the racing environment.

Why this exists: every experiment in this project is described by a TOML file.
A silently ignored typo (``max_sped = 1.0``) would waste a multi-minute training
run, so loading is strict: any key that is not a real dataclass field raises a
``ConfigError`` that names the offending key.

Units everywhere: meters, seconds, radians (unless a field name says otherwise).
"""

import dataclasses
import tomllib
import typing
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a config contains unknown keys or is otherwise invalid."""


@dataclass(frozen=True)
class CarParams:
    """Physical parameters of the kinematic-bicycle car model."""

    length: float = 4.5  # body length, m
    width: float = 2.0  # body width, m
    wheelbase: float = 2.7  # distance between axles, m
    max_steer: float = 0.6  # maximum front-wheel angle, rad
    steer_rate: float = 3.0  # how fast the wheel angle can change, rad/s
    max_accel: float = 8.0  # full-throttle acceleration, m/s^2
    max_brake: float = 16.0  # full-brake deceleration, m/s^2
    drag: float = 0.0015  # quadratic air drag coefficient, 1/m
    max_speed: float = 50.0  # hard speed cap, m/s
    max_lat_accel: float = 12.0  # grip limit: sideways acceleration before sliding, m/s^2


@dataclass(frozen=True)
class EnvConfig:
    """Everything that defines one racing environment (physics, sensing, rewards)."""

    dt: float = 1 / 60  # physics step, s (independent of rendering)
    action_repeat: int = 4  # physics steps per agent decision
    n_rays: int = 9  # number of distance-sensor rays
    ray_spread_deg: float = 180.0  # angle covered by the rays, degrees
    ray_range: float = 60.0  # max sensor distance, m
    progress_scale: float = 0.1  # reward per meter of track progress
    crash_penalty: float = 10.0  # reward lost on leaving the track
    time_cost: float = 0.01  # reward lost per agent step (encourages speed)
    lap_bonus: float = 10.0  # reward for completing a lap
    wrong_way_m: float = 15.0  # backwards progress (m) that ends the episode
    max_episode_s: float = 90.0  # episode time limit, s
    max_laps: int = 2  # laps that end the episode successfully
    random_start: bool = False  # start at a random point on the track
    car: CarParams = dataclasses.field(default_factory=CarParams)


def from_dict(cls, data: dict, _path: str = ""):
    """Build dataclass ``cls`` from a plain dict, rejecting unknown keys.

    Nested dataclass fields are built recursively, and error messages use a
    dotted path (``car.max_sped``) so the typo is easy to find in the TOML.
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        names = ", ".join(f"{_path}{k}" for k in unknown)
        raise ConfigError(f"unknown config key(s) for {cls.__name__}: {names}")

    # Field types are strings-free here (no `from __future__ import annotations`),
    # but get_type_hints keeps this robust if that ever changes.
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigError(f"{_path}{name} must be a table, got {type(value).__name__}")
            value = from_dict(hint, value, _path=f"{_path}{name}.")
        elif typing.get_origin(hint) is tuple and isinstance(value, list):
            # TOML has only arrays (lists); tuple-typed fields stay hashable/immutable.
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_toml(path: str | Path) -> dict:
    """Read a TOML file into a plain dict."""
    with open(path, "rb") as f:  # tomllib requires binary mode
        return tomllib.load(f)


def _merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` over ``base`` without mutating either."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_env_config(path: str | Path, overrides: dict | None = None) -> EnvConfig:
    """Load an EnvConfig from TOML, apply overrides, then validate strictly.

    Overrides are merged *before* validation so a typo in an override (for
    example from a command-line flag) is caught just like a typo in the file.
    """
    data = load_toml(path)
    if overrides:
        data = _merge(data, overrides)
    return from_dict(EnvConfig, data)
