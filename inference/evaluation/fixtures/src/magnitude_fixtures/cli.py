"""``magnitude-fixtures build <family> --output PATH``: write a fixture set as JSON Lines."""

import argparse
from pathlib import Path

from . import FAMILIES, build, write_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="magnitude-fixtures")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="build one family's fixture set")
    command.add_argument("family", choices=FAMILIES)
    command.add_argument("--output", type=Path, required=True, help="fixture set path (.jsonl)")
    command.add_argument("--count", type=int, default=8, help="independent sessions (lanes)")
    command.add_argument("--queries", type=int, default=4, help="retrieval keys per probe")
    args = parser.parse_args(argv)
    fixtures = build(args.family, count=args.count, queries=args.queries)
    print(f"{args.output}\t{len(fixtures)} fixtures\tsha256 {write_set(args.output, fixtures)}")
    return 0
