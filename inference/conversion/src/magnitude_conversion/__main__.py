"""magnitude-conversion: convert a DeepSeek V4.1 checkpoint to Magnitude GGUFs, and check a converted file.

  convert <checkpoint> <out.gguf> --part text|dspark|vision --source-repo R --source-revision H [--layers 0,1] [--split-max-size 50G]
  check   <checkpoint> <out.gguf or first shard> --part ... --source-repo R --source-revision H [--layers ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from . import check, deepseek41

logger = logging.getLogger("magnitude_conversion")


def _size(text: str) -> int:
    units = {"K": 1000, "M": 1000**2, "G": 1000**3}
    return int(text[:-1]) * units[text[-1]] if text and text[-1] in units else int(text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="magnitude-conversion")
    parser.add_argument("command", choices=("convert", "check"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--part", choices=deepseek41.PARTS, default="text")
    parser.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--source-repo", required=True, help="Hugging Face repository id of the checkpoint")
    parser.add_argument("--source-revision", required=True, help="its commit hash")
    parser.add_argument("--split-max-size", type=_size, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    url = f"https://huggingface.co/{args.source_repo}/tree/{args.source_revision}"
    with torch.inference_mode():
        model = deepseek41.build(args.part, args.checkpoint, args.gguf, source_repo_url=url, layers=args.layers,
                                 split_max_size=args.split_max_size)
        if args.command == "convert":
            model.write()
        else:
            checked = check.check(model, args.gguf)
            logger.info("byte-exact: %s", ", ".join(f"{k} {v / 1e9:.3f} GB" for k, v in sorted(checked.items())))


if __name__ == "__main__":
    main()
