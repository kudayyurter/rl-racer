"""Tests for run directories (racer/runlog.py) and policy loading (racer/policy.py)."""

import csv
import re

import numpy as np
import pytest
import torch

from racer import policy as policy_mod
from racer.policy import load_policy, register_loader
from racer.runlog import METRIC_COLUMNS, RunDir, latest_checkpoint


def metric_row(i: int) -> dict:
    return {c: float(i) + k for k, c in enumerate(METRIC_COLUMNS)}


def make_ckpt(iteration: int, method: str = "evolution") -> dict:
    return {"method": method, "iteration": iteration, "env_steps": 100 * iteration,
            "obs_dim": 12, "arch": {"hidden": [8]}, "weights": np.arange(3.0) + iteration,
            "env_config": {"max_laps": 2}}


def test_create_names_dir_and_avoids_collisions(tmp_path):
    a = RunDir.create("evolution", 3, root=tmp_path)
    b = RunDir.create("evolution", 3, root=tmp_path)  # same second, same method and seed
    assert re.fullmatch(r"evolution-\d{8}-\d{6}-s3", a.path.name)
    assert a.path != b.path and a.path.is_dir() and b.path.is_dir()


def test_save_config(tmp_path):
    run = RunDir.create("x", 0, root=tmp_path)
    run.save_config({"lr": 0.1})
    assert (run.path / "config.json").read_text().strip().startswith("{")


def test_metrics_round_trip(tmp_path):
    run = RunDir.create("x", 0, root=tmp_path)
    run.log(metric_row(1))
    run.log(metric_row(2))
    lines = (run.path / "metrics.csv").read_text().splitlines()
    assert lines[0].split(",") == METRIC_COLUMNS
    assert sum(line.startswith("iteration") for line in lines) == 1  # header once
    rows = run.read_metrics()
    assert rows == [metric_row(1), metric_row(2)]


def test_log_rejects_missing_and_unknown_columns(tmp_path):
    run = RunDir.create("x", 0, root=tmp_path, extra_columns=("entropy",))
    with pytest.raises(ValueError, match="eval_return"):
        run.log({c: 0.0 for c in METRIC_COLUMNS if c != "eval_return"})
    with pytest.raises(ValueError, match="entropi"):
        run.log({**metric_row(1), "entropi": 1.0})
    run.log(metric_row(1))  # a missing extra column is allowed (written empty)
    run.log({**metric_row(2), "entropy": 0.5})
    assert run.read_metrics()[1]["entropy"] == 0.5
    assert run.read_metrics()[0]["entropy"] != run.read_metrics()[0]["entropy"]  # NaN


def test_open_recovers_columns_from_header(tmp_path):
    run = RunDir.create("x", 0, root=tmp_path, extra_columns=("entropy",))
    run.log({**metric_row(1), "entropy": 0.5})
    reopened = RunDir.open(run.path)
    reopened.log({**metric_row(2), "entropy": 0.25})
    with open(run.path / "metrics.csv") as f:
        assert next(csv.reader(f)) == METRIC_COLUMNS + ["entropy"]
    assert [r["entropy"] for r in reopened.read_metrics()] == [0.5, 0.25]


def test_checkpoints_written_with_latest(tmp_path):
    run = RunDir.create("x", 0, root=tmp_path)
    run.save_checkpoint(make_ckpt(1), 1)
    run.save_checkpoint(make_ckpt(2), 2)
    assert sorted(p.name for p in (run.path / "checkpoints").iterdir()) == [
        "ckpt_00001.pt", "ckpt_00002.pt", "latest.pt"]
    # weights_only=False: our own checkpoints hold numpy arrays.
    assert torch.load(latest_checkpoint(run.path), weights_only=False)["iteration"] == 2


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    run = RunDir.create("x", 0, root=tmp_path)
    run.save_checkpoint(make_ckpt(1), 1)

    def boom_after_partial_write(obj, f):
        with open(f, "wb") if not hasattr(f, "write") else f as fh:
            fh.write(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom_after_partial_write)
    with pytest.raises(RuntimeError):
        run.save_checkpoint(make_ckpt(2), 2)
    monkeypatch.undo()
    assert torch.load(latest_checkpoint(run.path), weights_only=False)["iteration"] == 1
    # No partial file left behind, temp or otherwise.
    assert sorted(p.name for p in (run.path / "checkpoints").iterdir()) == [
        "ckpt_00001.pt", "latest.pt"]


def test_latest_checkpoint_empty_run(tmp_path):
    assert latest_checkpoint(RunDir.create("x", 0, root=tmp_path).path) is None


def test_ghost_round_trip_sorted(tmp_path):
    run = RunDir.create("x", 0, root=tmp_path)
    t, poses = np.linspace(0, 5, 4), np.random.default_rng(0).normal(size=(4, 3))
    run.save_ghost(10, t, poses, 5.0)
    run.save_ghost(2, t * 2, poses + 1, 10.0)
    ghosts = run.ghosts()
    assert [g.iteration for g in ghosts] == [2, 10]
    np.testing.assert_array_equal(ghosts[1].poses, poses)
    np.testing.assert_array_equal(ghosts[0].t, t * 2)
    assert ghosts[0].lap_time == 10.0
    assert not [p for p in (run.path / "ghosts").iterdir() if p.suffix != ".npz"]


def test_load_policy_dispatches_to_registered_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_mod, "_LOADERS", dict(policy_mod._LOADERS))
    run = RunDir.create("x", 0, root=tmp_path)
    run.save_checkpoint(make_ckpt(4, method="fake"), 4)
    register_loader("fake", lambda ckpt: ("loaded", ckpt["iteration"]))
    assert load_policy(latest_checkpoint(run.path)) == ("loaded", 4)


def test_load_policy_unknown_method(tmp_path):
    run = RunDir.create("x", 0, root=tmp_path)
    run.save_checkpoint(make_ckpt(1, method="nonsense"), 1)
    with pytest.raises(ValueError, match="nonsense"):
        load_policy(latest_checkpoint(run.path))
