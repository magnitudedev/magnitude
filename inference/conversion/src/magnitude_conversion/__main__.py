"""magnitude-conversion: convert a DeepSeek V4.1 checkpoint to Magnitude GGUFs, check a converted file against it, and
verify that a file alone does what its users need before it is published.

  convert <checkpoint> <out.gguf> --part text|dspark|vision --source-repo R --source-revision H [--layers 0,1] [--pool 8,2,8] [--split-max-size 50G] [--shards 1]
  check   <checkpoint> <out.gguf or first shard> --part ... --source-repo R --source-revision H [--layers ...] [--pool ...]
  verify-publish <out.gguf or first shard>

`--pool BLOCKS,BLOCK_ROWS,TOP_K` rewrites the indexer's candidate pool and top-k: a test definition whose pool binds
over short prompts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from . import check, deepseek41, verify

logger = logging.getLogger("magnitude_conversion")


def _size(text: str) -> int:
    units = {"K": 1000, "M": 1000**2, "G": 1000**3}
    return int(text[:-1]) * units[text[-1]] if text and text[-1] in units else int(text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="magnitude-conversion")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("convert", "check"):
        command = commands.add_parser(name)
        command.add_argument("checkpoint", type=Path)
        command.add_argument("gguf", type=Path)
        command.add_argument("--part", choices=deepseek41.PARTS, default="text")
        command.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")])
        command.add_argument("--pool", type=lambda s: tuple(int(x) for x in s.split(",")),
                             help="BLOCKS,BLOCK_ROWS,TOP_K of the indexer's candidate pool")
        command.add_argument("--source-repo", required=True, help="Hugging Face repository id of the checkpoint")
        command.add_argument("--source-revision", required=True, help="its commit hash")
    convert = commands.choices["convert"]
    convert.add_argument("--split-max-size", type=_size, default=0)
    convert.add_argument("--shards", type=lambda s: frozenset(int(x) for x in s.split(",")),
                         help="write only these shards of the split file, numbered from 1")
    # check reads the file's shards as they are: it writes nothing, so neither option applies
    commands.choices["check"].set_defaults(split_max_size=0, shards=None)
    commands.add_parser("verify-publish").add_argument("gguf", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "verify-publish":
        try:
            lines = verify.verify(args.gguf)
        except verify.Refused as refused:
            print(f"{refused}\nverify-publish: refused {args.gguf}", flush=True)
            sys.exit(1)
        print("\n".join(lines) + f"\nverify-publish: pass {args.gguf}", flush=True)
        return

    url = f"https://huggingface.co/{args.source_repo}/tree/{args.source_revision}"
    with torch.inference_mode():
        model = deepseek41.build(args.part, args.checkpoint, args.gguf, source_repo_url=url, layers=args.layers,
                                 pool=args.pool, split_max_size=args.split_max_size,
                                 shards=args.shards)
        if args.command == "convert":
            model.write()
        else:
            checked = check.check(model, args.gguf)
            logger.info("byte-exact: %s", ", ".join(f"{k} {v / 1e9:.3f} GB" for k, v in sorted(checked.items())))


if __name__ == "__main__":
    main()
