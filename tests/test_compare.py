"""Tests for racer.compare: grouping, success statistics, curve aggregation, outputs."""

import math

import numpy as np
import pytest

from racer.cli import main
from racer.compare import aggregate, load_run, summarize, summary_markdown
from racer.runlog import RunDir

NAN = float("nan")


def make_run(root, method, seed, rows, track="twisty", env=None, train=None):
    """Fake run dir. rows: (env_steps, wall_s, eval_progress, eval_laps, best_lap) tuples;
    eval_progress None means a training-only row (NaN eval columns). ``train`` adds method
    settings to the train config, which (like a real config.json) also records the seed."""
    run = RunDir.create(method, seed, root=root)
    run.save_config({"method": method, "seed": seed,
                     "train": {"track": track, "seed": seed, **(train or {})},
                     "env": env if env is not None else {"dt": 0.0167}})
    for i, (steps, wall, prog, laps, best) in enumerate(rows, start=1):
        row = dict(iteration=i, env_steps=steps, wall_time_s=wall, train_return_mean=0.0,
                   eval_return=NAN if prog is None else prog, eval_progress_m=NAN if prog is None else prog,
                   eval_laps=NAN if prog is None else laps, eval_best_lap_s=best)
        run.log(row)
    return run.path


def three_seeds(tmp_path):
    a = make_run(tmp_path, "ppo", 0, [(100, 1, None, 0, NAN), (200, 2, 50.0, 0, NAN),
                                      (300, 3, 400.0, 1, 30.0), (400, 4, 500.0, 2, 28.0)])
    b = make_run(tmp_path, "ppo", 1, [(100, 1, 20.0, 0, NAN), (300, 4, 450.0, 1, 34.0),
                                      (500, 7, 500.0, 1, 32.0)])
    c = make_run(tmp_path, "ppo", 2, [(100, 1, 10.0, 0, NAN), (500, 5, 80.0, 0, NAN)])
    return [a, b, c]


def test_summarize_groups_and_success(tmp_path):
    rows = summarize(three_seeds(tmp_path))
    assert len(rows) == 1
    r = rows[0]
    assert (r["method"], r["track"], r["seeds"], r["success"]) == ("ppo", "twisty", 3, "2/3")
    # Per-seed best laps are 28 and 32; median over the two successful seeds.
    assert r["best_lap_median"] == pytest.approx(30.0)
    assert (r["best_lap_min"], r["best_lap_max"]) == (28.0, 32.0)
    # First lap at steps 300 / 300 and wall 3 / 4.
    assert r["steps_to_lap"] == pytest.approx(300)
    assert r["time_to_lap"] == pytest.approx(3.5)


def test_best_lap_so_far_is_running_min_over_eval_rows_only(tmp_path):
    p = make_run(tmp_path, "ppo", 0, [(10, 1, None, 0, NAN), (20, 2, 5.0, 1, 40.0),
                                      (30, 3, 6.0, 1, 45.0), (40, 4, 7.0, 1, 38.0)])
    c = load_run(p)
    assert list(c["steps"]) == [20, 30, 40]            # the training-only row is dropped
    assert list(c["best_lap"]) == [40.0, 40.0, 38.0]   # never gets worse


def test_groups_split_by_track_and_env_hash(tmp_path):
    runs = [make_run(tmp_path, "ppo", 0, [(1, 1, 1.0, 0, NAN)], track="twisty"),
            make_run(tmp_path, "ppo", 1, [(1, 1, 1.0, 0, NAN)], track="oval"),
            make_run(tmp_path, "ppo", 2, [(1, 1, 1.0, 0, NAN)], track="oval", env={"dt": 0.02}),
            make_run(tmp_path, "a2c", 0, [(1, 1, 1.0, 0, NAN)], track="oval")]
    rows = summarize(runs)
    assert len(rows) == 4
    labels = [r["label"] for r in rows]
    assert len(set(labels)) == 4
    # Hash suffix only where method+track collide.
    assert sum("#" in lab for lab in labels) == 2
    assert "ppo · twisty" in labels and "a2c · oval" in labels


def test_groups_split_by_train_config_but_not_by_seed(tmp_path):
    # Seeds 0 and 1 of the same settings are one group; a different mutation_std (or
    # population, budget, ...) on the same track and env is a different experiment.
    same = [make_run(tmp_path, "evolution", s, [(1, 1, 1.0, 0, NAN)], train={"mutation_std": 0.1})
            for s in (0, 1)]
    other = make_run(tmp_path, "evolution", 2, [(1, 1, 1.0, 0, NAN)], train={"mutation_std": 0.5})
    rows = summarize([*same, other])
    assert sorted(r["seeds"] for r in rows) == [1, 2]
    labels = [r["label"] for r in rows]
    assert len(set(labels)) == 2
    # Same env, so the suffix that tells them apart is the train-config hash.
    assert all("train#" in lab for lab in labels)


def test_run_with_no_eval_rows_counts_as_failed_seed(tmp_path):
    runs = [make_run(tmp_path, "evolution", 0, [(1, 1, None, 0, NAN)]),
            make_run(tmp_path, "evolution", 1, [(1, 1, 5.0, 0, NAN)])]
    r = summarize(runs)[0]
    assert r["success"] == "0/2"
    assert math.isnan(r["best_lap_median"])


def test_aggregate_steps_values_and_never_extrapolates(tmp_path):
    curves = [load_run(p) for p in three_seeds(tmp_path)]
    x, med, lo, hi, n = aggregate(curves, "steps", "progress")
    assert list(x) == [100, 200, 300, 400, 500]
    # At x=200: seed a=50, b holds 20 (from 100), c holds 10 -> median 20, band 10..50.
    assert (med[1], lo[1], hi[1]) == (20.0, 10.0, 50.0)
    # At x=400: a is at its last point (500, still defined), b holds 450, c holds 10.
    assert med[3] == 450.0
    # At x=500 run a has ended: only b (500) and c (80) are used, not an extrapolated a.
    assert (lo[4], hi[4]) == (80.0, 500.0)
    # Contributing seeds per x: a has no eval point at x=100 and ends after x=400.
    assert list(n) == [2, 3, 3, 3, 2]


def test_aggregate_best_lap_ignores_seeds_without_a_lap_yet(tmp_path):
    curves = [load_run(p) for p in three_seeds(tmp_path)]
    x, med, lo, hi, n = aggregate(curves, "steps", "best_lap")
    assert math.isnan(med[0])                  # nobody has lapped at x=100
    assert (lo[2], hi[2]) == (30.0, 34.0)      # seed c has no lap and is left out
    assert med[2] == 32.0
    # Seed a laps at x=300, b at x=300; c never: counts 0,0,2,2,1 (a ended after x=400).
    assert list(n) == [0, 0, 2, 2, 1]


def test_cli_compare_writes_plots_and_summary(tmp_path):
    runs = three_seeds(tmp_path / "runs")
    out = tmp_path / "results"
    assert main(["compare", *map(str, runs), "--out", str(out), "--name", "t"]) == 0
    assert (out / "t_vs_steps.png").stat().st_size > 1000
    assert (out / "t_vs_time.png").stat().st_size > 1000
    text = (out / "summary.md").read_text()
    assert "## t" in text and "| ppo · twisty | 3 | 2/3 |" in text
    assert "30.00 (28.00–32.00)" in text
    # A second invocation appends rather than overwrites.
    main(["compare", *map(str, runs), "--out", str(out), "--name", "u"])
    assert (out / "summary.md").read_text().count("## ") == 2


def test_summary_table_tells_clashing_groups_apart(tmp_path):
    runs = [make_run(tmp_path, "ppo", 0, [(1, 1, 1.0, 0, NAN)], track="oval"),
            make_run(tmp_path, "ppo", 1, [(1, 1, 1.0, 0, NAN)], track="oval", env={"dt": 0.02})]
    rows = summarize(runs)
    text = summary_markdown(rows, "x")
    lines = [ln for ln in text.splitlines() if ln.startswith("| ppo")]
    assert len(lines) == 2 and lines[0].split("|")[1] != lines[1].split("|")[1]
    assert all("#" in ln.split("|")[1] for ln in lines)


def test_aggregate_with_no_eval_rows_is_empty(tmp_path):
    curves = [load_run(make_run(tmp_path, "evolution", 0, [(1, 1, None, 0, NAN)]))]
    x, med, lo, hi, n = aggregate(curves, "steps", "progress")
    assert len(x) == len(med) == len(n) == 0


def test_cli_compare_on_run_stopped_before_first_eval(tmp_path):
    # A run killed before its first eval row must not crash the plots.
    run = make_run(tmp_path / "runs", "evolution", 0, [(1, 1, None, 0, NAN)])
    out = tmp_path / "results"
    assert main(["compare", str(run), "--out", str(out), "--name", "t"]) == 0
    assert "| evolution · twisty | 1 | 0/1 |" in (out / "summary.md").read_text()
