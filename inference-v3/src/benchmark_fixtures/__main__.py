"""Prepare a fixture in the shared local cache without loading model weights."""

import argparse
import asyncio
import json
from pathlib import Path

from .preparation import Fixture, Tokenization, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", choices=("prose.moby-dick", "tools.bfcl"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--continuation", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    fixture = Fixture(
        identity=args.fixture,
        context_tokens=args.context,
        continuation_tokens=args.continuation,
        offset=args.offset,
    )
    prepared = asyncio.run(prepare(fixture, Tokenization(args.artifact)))
    print(json.dumps(prepared.provenance, indent=2))


if __name__ == "__main__":
    main()
