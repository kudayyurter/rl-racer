"""Command-line entry point: the ``racer`` console script.

Each subcommand is a stub for now; later tasks replace the stubs with real
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="racer", description="RL Racer")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text).set_defaults(handler=_not_implemented)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
