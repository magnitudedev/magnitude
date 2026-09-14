"""Device-free formula measurement history browser."""

import argparse
from pathlib import Path

from .archive import browse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="existing formula observation SQLite store")
    args = parser.parse_args()
    if not args.store.is_file():
        parser.error("observation store does not exist")
    browse(args.store)


if __name__ == "__main__":
    main()
