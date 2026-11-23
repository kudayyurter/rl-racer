"""Tests for strict TOML config loading: defaults, overrides, and typo detection."""

import dataclasses

import pytest

from racer.cli import main
from racer.config import ConfigError, EnvConfig, from_dict, load_env_config


def test_defaults_load():
    cfg = load_env_config("configs/env.toml")
    assert cfg.n_rays == 9 and cfg.car.max_speed == 50.0 and cfg.action_repeat == 4


def test_overrides_apply():
    cfg = load_env_config("configs/env.toml", {"time_cost": 0.05, "car": {"max_lat_accel": 9.0}})
    assert cfg.time_cost == 0.05 and cfg.car.max_lat_accel == 9.0 and cfg.car.max_speed == 50.0


def test_unknown_key_raises():
    with pytest.raises(ConfigError, match="mutation_sdt"):
        from_dict(EnvConfig, {"mutation_sdt": 0.1})


def test_unknown_nested_key_raises():
    with pytest.raises(ConfigError, match="car.max_sped"):
        from_dict(EnvConfig, {"car": {"max_sped": 1.0}})


def test_list_becomes_tuple_for_tuple_fields():
    @dataclasses.dataclass(frozen=True)
    class Net:
        hidden: tuple[int, ...] = (64, 64)

    assert from_dict(Net, {"hidden": [32, 16]}).hidden == (32, 16)


def test_cli_stub_returns_one(capsys):
    # Any subcommand that is still a stub (drive became real in Task 7).
    assert main(["compare"]) == 1
    assert "not implemented yet" in capsys.readouterr().out
