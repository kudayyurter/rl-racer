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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
