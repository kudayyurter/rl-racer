"""Command-line entry point: the ``racer`` console script.

Every subcommand imports its implementation lazily inside its handler, so
``racer --help`` stays fast and does not load raylib, torch or matplotlib.
"""

import argparse
import importlib
import sys

COMMANDS = {
    "drive": "play the game yourself with the keyboard",
    "train": "train an agent",
    "watch": "watch a trained agent drive",
    "compare": "compare trained agents head to head",
    "bench": "benchmark simulation speed",
    "gen-track": "generate a random track",
}


def _error(command: str, message: str) -> int:
    """A user mistake (typo in a key, a track or a path): one line on stderr, exit code 2."""
    print(f"racer {command}: {message}", file=sys.stderr)
    return 2


def _missing_track(name: str) -> str | None:
    """None if ``name`` loads as a track, else a one-line reason (without loading raylib)."""
    from racer.track import load_track

    try:
        load_track(name)
    except FileNotFoundError as e:
        return f"no track {name!r}: {e.filename} does not exist"
    return None


def _bench(args: argparse.Namespace) -> int:
    # Imported here so `racer --help` stays fast and doesn't load the simulator.
    from racer.bench import bench, format_table

    print(format_table(bench(seconds=args.seconds)))
    return 0


def _drive(args: argparse.Namespace) -> int:
    # Imported here so `racer --help` doesn't load raylib.
    from racer.drive import run_drive

    return run_drive(args.track, overlay=args.overlay, screenshot=args.screenshot,
                     frames=args.frames)


def _watch(args: argparse.Namespace) -> int:
    # Checked before racer.watch is imported: a typo should not open a window first.
    if args.track is not None and (problem := _missing_track(args.track)):
        return _error("watch", problem)
    # Imported here so `racer --help` doesn't load raylib or torch.
    from racer.watch import run_watch

    return run_watch(args.run_dir, args.track, reload_s=args.reload_s, n_ghosts=args.ghosts,
                     screenshot=args.screenshot, frames=args.frames)


def _compare(args: argparse.Namespace) -> int:
    # Imported here so `racer --help` doesn't load matplotlib.
    from racer.compare import run_compare

    return run_compare(args.run_dirs, args.out, args.name)


def _gen_track(args: argparse.Namespace) -> int:
    from racer.trackgen import write_track_json

    path = write_track_json(args.seed, args.out)
    print(f"wrote {path}")
    return 0


# One line per learning method: name -> module. Each module defines CONFIG_CLASS (its config
# dataclass), EXTRA_COLUMNS (extra metrics.csv columns) and train(cfg, env_cfg, run).
TRAIN_METHODS = {
    "evolution": "racer.algos.evolution",
    "reinforce": "racer.algos.reinforce",
    "a2c": "racer.algos.a2c",
    "ppo": "racer.algos.ppo",
    "sb3": "racer.algos.sb3_ppo",
}


def _train(args: argparse.Namespace) -> int:
    # Imported here so `racer --help` doesn't load numpy-heavy training code.
    from racer.config import ConfigError, load_env_config, load_method_config, parse_overrides
    from racer.runlog import RunDir

    module = importlib.import_module(TRAIN_METHODS[args.method])
    # Every user mistake is caught here, before a run directory exists, so a typo leaves
    # neither a traceback nor an empty runs/ entry behind.
    try:
        overrides = parse_overrides(args.set)
        if args.seed is not None:
            overrides["seed"] = args.seed
        cfg = load_method_config(module.CONFIG_CLASS, args.config or f"configs/{args.method}.toml",
                                 overrides)
        env_cfg = load_env_config(args.env_config, parse_overrides(args.env_set))
    except ConfigError as e:
        return _error("train", f"bad config: {e}")
    except FileNotFoundError as e:
        return _error("train", f"no config file {e.filename}")
    if problem := _missing_track(cfg.track):
        return _error("train", problem)
    run = RunDir.create(args.method, cfg.seed, extra_columns=module.EXTRA_COLUMNS)
    print(f"run directory: {run.path}")
    module.train(cfg, env_cfg, run)
    return 0


def _add_train_parser(train: argparse.ArgumentParser) -> None:
    methods = train.add_subparsers(dest="method", required=True)
    for name in TRAIN_METHODS:
        p = methods.add_parser(name, help=f"train with {name}")
        p.add_argument("--config", help=f"method config TOML (default configs/{name}.toml)")
        p.add_argument("--env-config", default="configs/env.toml",
                       help="environment config TOML (default configs/env.toml)")
        p.add_argument("--seed", type=int, help="random seed (default: the config's seed)")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a method-config value, e.g. --set track=oval (repeatable)")
        p.add_argument("--env-set", action="append", default=[], metavar="KEY=VALUE",
                       help="override an env-config value, e.g. --env-set car.max_lat_accel=9")
    train.set_defaults(handler=_train)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="racer", description="RL Racer")
    sub = parser.add_subparsers(dest="command", required=True)
    subparsers = {}
    for name, help_text in COMMANDS.items():
        subparsers[name] = sub.add_parser(name, help=help_text)

    subparsers["bench"].add_argument("--seconds", type=float, default=3.0,
                                     help="wall time per batch size (default 3)")
    subparsers["bench"].set_defaults(handler=_bench)

    drive = subparsers["drive"]
    drive.add_argument("--track", default="twisty",
                       help="track name (tracks/NAME.json) or path to a track JSON (default twisty)")
    drive.add_argument("--overlay", action="store_true",
                       help="start with the debug overlay on (D toggles it)")
    drive.add_argument("--screenshot", metavar="PATH",
                       help="render --frames frames, save a PNG to PATH, and exit")
    drive.add_argument("--frames", type=int, default=30,
                       help="frames to render before the screenshot (default 30)")
    drive.set_defaults(handler=_drive)

    watch = subparsers["watch"]
    watch.add_argument("run_dir", metavar="RUN_DIR", help="run directory, e.g. runs/evolution-...")
    watch.add_argument("--track", help="track to drive on (default: the track the run trains on)")
    watch.add_argument("--reload-s", type=float, default=3.0,
                       help="seconds between checks for a new checkpoint (default 3)")
    watch.add_argument("--ghosts", type=int, default=5,
                       help="how many of the latest ghost laps to replay (default 5)")
    watch.add_argument("--screenshot", metavar="PATH",
                       help="render --frames frames, save a PNG to PATH, and exit")
    watch.add_argument("--frames", type=int, default=30,
                       help="frames to render before the screenshot (default 30)")
    watch.set_defaults(handler=_watch)

    compare = subparsers["compare"]
    compare.add_argument("run_dirs", nargs="+", metavar="RUN_DIR", help="finished run directories")
    compare.add_argument("--out", default="results", help="output directory (default results)")
    compare.add_argument("--name", default="compare",
                         help="file prefix and summary heading (default compare)")
    compare.set_defaults(handler=_compare)

    gen = subparsers["gen-track"]
    gen.add_argument("--seed", type=int, required=True, help="seed: the same seed gives the same track")
    gen.add_argument("--out", required=True, metavar="PATH", help="where to write the track JSON, e.g. tracks/rand_7.json")
    gen.set_defaults(handler=_gen_track)

    _add_train_parser(subparsers["train"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
