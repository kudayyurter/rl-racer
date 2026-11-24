"""Tests for strict TOML config loading: defaults, overrides, and typo detection."""

import dataclasses

import pytest

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


# ---- value type checking (overrides come from the command line as untyped text)

from racer.config import parse_overrides  # noqa: E402


@pytest.mark.parametrize("override", ["random_start=False", "dt=abc", "n_rays=2.5",
                                      "n_rays=true", "car.drag=\"x\""])
def test_env_override_with_wrong_type_raises(override):
    with pytest.raises(ConfigError, match=override.split("=")[0].split(".")[-1]):
        from_dict(EnvConfig, parse_overrides([override]))


def test_int_accepted_for_float_field():
    assert from_dict(EnvConfig, parse_overrides(["dt=1"])).dt == 1


def test_tuple_field_type_checked():
    from racer.algos.evolution import EvolutionConfig
    base = {"env_steps_budget": 1}
    with pytest.raises(ConfigError, match="hidden"):
        from_dict(EvolutionConfig, {**base, **parse_overrides(["hidden=64"])})
    with pytest.raises(ConfigError, match="hidden"):
        from_dict(EvolutionConfig, {**base, **parse_overrides(["hidden=[8, \"a\"]"])})
    assert from_dict(EvolutionConfig, {**base, "hidden": [8, 4]}).hidden == (8, 4)


def test_shipped_configs_load():
    from racer.algos.evolution import EvolutionConfig
    from racer.config import load_env_config, load_method_config
    load_env_config("configs/env.toml")
    load_method_config(EvolutionConfig, "configs/evolution.toml")


def test_env_fastlap_differs_from_env_only_in_time_cost_and_time_limit():
    base = dataclasses.asdict(load_env_config("configs/env.toml"))
    fast = dataclasses.asdict(load_env_config("configs/env_fastlap.toml"))
    changed = {k for k in base if base[k] != fast[k]}
    assert changed == {"time_cost", "max_episode_s"}
    assert fast["time_cost"] == pytest.approx(5 * base["time_cost"])


def test_env_fastlap_laps_finish_inside_time_limit():
    # The time cost can only pay for speed if a faster car ends its episode sooner: every
    # driver lapping in 60 s or less must finish max_laps before the time limit cuts it off.
    fast = load_env_config("configs/env_fastlap.toml")
    assert fast.max_laps * 60.0 <= fast.max_episode_s
