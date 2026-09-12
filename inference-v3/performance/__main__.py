"""Manage Magnitensor-backed inference performance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from performance.session import ingest
from performance.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_command = commands.add_parser("import", help="import a completed session benchmark")
    import_command.add_argument("directory", type=Path)
    import_command.add_argument("--store", type=Path, default=Path("runs/performance"))
    args = parser.parse_args()
    if args.command == "import":
        result = ingest(args.directory.expanduser().resolve(strict=True), Store(args.store))
        print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
