"""A mistyped command line gets one clear line and a non-zero exit, not a traceback."""

from pathlib import Path

from racer.cli import main
from racer.runlog import RunDir

RUNS = Path("runs")


def _run_dirs() -> set[str]:
    return {p.name for p in RUNS.iterdir()} if RUNS.is_dir() else set()


def test_train_with_unknown_setting_prints_one_line(capsys):
    before = _run_dirs()
    assert main(["train", "evolution", "--set", "mutaton_std=0.5"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("racer train: ") and "mutaton_std" in err
    assert err.count("\n") == 1
    assert _run_dirs() == before   # refused before a run directory was made


def test_train_with_unknown_track_prints_one_line(capsys):
    before = _run_dirs()
    assert main(["train", "ppo", "--set", "track=twsty"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("racer train: ") and "twsty" in err
    assert err.count("\n") == 1
    assert _run_dirs() == before


def test_train_with_missing_config_file_prints_one_line(capsys):
    assert main(["train", "a2c", "--config", "configs/nope.toml"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("racer train: ") and "configs/nope.toml" in err


def test_watch_with_unknown_track_prints_one_line(tmp_path, capsys):
    run = RunDir.create("evolution", 0, root=tmp_path)
    assert main(["watch", str(run.path), "--track", "ovl"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("racer watch: ") and "ovl" in err
    assert err.count("\n") == 1


def test_train_with_directory_or_bad_json_track_prints_one_line(tmp_path, capsys):
    before = _run_dirs()
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "bad", "points": [[0, 0], [1, 0], [1, 1]]}')   # no width
    for track in (f"{tmp_path}/", str(bad)):
        assert main(["train", "ppo", "--set", f"track={track}"]) == 2
        err = capsys.readouterr().err
        assert err.startswith("racer train: ") and err.count("\n") == 1
    assert _run_dirs() == before
