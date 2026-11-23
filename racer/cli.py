"""Command-line entry point: the ``racer`` console script.

Subcommands not built yet are stubs; later tasks replace the stubs with real
implementations. Keeping the full command list here from the start means the
CLI surface is visible (and ``racer --help`` works) from day one.
"""

import argparse

COMMANDS = {
    "drive": "play the game yourself with the keyboard",
    "train": "train an agent",
    "watch": "watch a trained agent drive",
    "compare": "compare trained agents head to head",
    "bench": "benchmark simulation speed",
    "gen-track": "generate a random track",
}


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"racer {args.command}: not implemented yet")
    return 1


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="racer", description="RL Racer")
    sub = parser.add_subparsers(dest="command", required=True)
    subparsers = {}
    for name, help_text in COMMANDS.items():
        subparsers[name] = sub.add_parser(name, help=help_text)
        subparsers[name].set_defaults(handler=_not_implemented)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
