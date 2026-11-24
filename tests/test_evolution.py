"""Tests for neuroevolution (racer/algos/evolution.py) and the `train` CLI plumbing."""

import dataclasses
import math

import numpy as np
import pytest

from racer.algos.evolution import EXTRA_COLUMNS, EvolutionConfig, train_evolution
from racer.cli import build_parser
from racer.config import ConfigError, EnvConfig, from_dict, parse_overrides
from racer.policy import load_policy
from racer.runlog import RunDir

# One-second episodes: no car can crash in that time, so every generation costs exactly
# population * 15 decisions and a fixed genome always scores the same.
SHORT = EnvConfig(max_episode_s=1.0)


def _run(tmp_path, generations, **cfg_kw):
    cfg = EvolutionConfig(track="oval", population=8, elite=2, hidden=(8,),
                          env_steps_budget=8 * 15 * generations - 1, **cfg_kw)
    run = RunDir.create("evolution", cfg.seed, root=tmp_path, extra_columns=EXTRA_COLUMNS)
    train_evolution(cfg, SHORT, run)
    return run


def test_evolution_smoke(tmp_path):
    run = _run(tmp_path, 2)
    rows = run.read_metrics()
    assert len(rows) == 2
    assert rows[0]["env_steps"] == 8 * 15
    assert math.isnan(rows[0]["eval_laps"]) and not math.isnan(rows[1]["eval_laps"])
    policy = load_policy(run.path / "checkpoints" / "latest.pt")
    assert policy.act(np.zeros((1, 12))).shape == (1, 2)
    assert [a.shape for a in policy.activations(np.zeros((1, 12)))] == [(1, 12), (1, 8), (1, 2)]


def test_elites_survive(tmp_path):
    rows = _run(tmp_path, 3).read_metrics()
    assert len(rows) == 3
    best = [r["best_fitness"] for r in rows]
    assert best[1] >= best[0] and best[2] >= best[1]


def test_same_seed_same_run(tmp_path):
    a = _run(tmp_path / "a", 2).read_metrics()
    b = _run(tmp_path / "b", 2).read_metrics()
    assert [r["best_fitness"] for r in a] == [r["best_fitness"] for r in b]


def test_parse_overrides():
    assert parse_overrides(["car.max_lat_accel=9", "track=oval", "hidden=[16, 16]", "x=true"]) == {
        "car": {"max_lat_accel": 9}, "track": "oval", "hidden": [16, 16], "x": True}
    with pytest.raises(ConfigError):
        parse_overrides(["novalue"])


def test_override_typo_fails_strict_loader():
    with pytest.raises(ConfigError, match="mutaton_std"):
        from_dict(EvolutionConfig, {"env_steps_budget": 1, **parse_overrides(["mutaton_std=1"])})
    assert dataclasses.is_dataclass(from_dict(EnvConfig, parse_overrides(["car.drag=0.01"])))


def test_train_cli_parses_shared_flags():
    args = build_parser().parse_args(["train", "evolution", "--seed", "3", "--set", "track=oval",
                                      "--set", "population=8", "--env-set", "car.drag=0.01"])
    assert (args.method, args.seed, args.set, args.env_set) == (
        "evolution", 3, ["track=oval", "population=8"], ["car.drag=0.01"])
