"""magnitude-conversion: convert a checkpoint to Magnitude GGUFs, check a converted file against it, and verify that a
file alone does what its users need before it is published. The family is the checkpoint's `config.json`
`model_type`: DeepSeek V4.1 (`deepseek_v41`, parts text, dspark and vision) or GLM-5.3-Flash (`glm5_next`, Z.ai's
FP8 release or NVIDIA's ModelOpt one, parts text and vision).

  convert <checkpoint> <out.gguf> --part text|dspark|vision --source-repo R --source-revision H [--layers 0,1] [--pool 8,2,8] [--split-max-size 50G] [--shards 1|''] [--upstream-f32]
  check   <checkpoint> <out.gguf or first shard> --part ... --source-repo R --source-revision H [--layers ...] [--pool ...]
  verify-publish <out.gguf or first shard>
  llama-logits <model.gguf> <tokens.npy> <logits.npy>

`--pool BLOCKS,BLOCK_ROWS,TOP_K` (DeepSeek V4.1) rewrites the indexer's candidate pool and top-k: a test definition
whose pool binds over short prompts. `--upstream-f32` (GLM-5.3-Flash) writes upstream's own conversion with every
tensor F32 instead, for upstream llama.cpp to run (`glm5next.upstream_f32`); it is never published.

A `--layers` slice's provenance record is catalog-tool's to write (`magnitude-service-catalog-tool slice-record
--source REPO@REVISION/SHARD... --layers L,... OUT.gguf`), the one writer of the format.

`llama-logits` writes upstream llama.cpp's float32 logits [prompts, length, vocabulary] of int64 tokens [prompts,
length] (`llama.logits`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from . import check, deepseek41, glm5next, llama, verify

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
        command.add_argument("--part", default="text", help="text, vision, or (DeepSeek V4.1) dspark")
        command.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")])
        command.add_argument("--pool", type=lambda s: tuple(int(x) for x in s.split(",")),
                             help="BLOCKS,BLOCK_ROWS,TOP_K of the indexer's candidate pool")
        command.add_argument("--source-repo", required=True, help="Hugging Face repository id of the checkpoint")
        command.add_argument("--source-revision", required=True, help="its commit hash")
        command.add_argument("--shards", type=lambda s: frozenset(int(x) for x in s.split(",") if x),
                             help=f"{name} only these shards of the split file, numbered from 1"
                             + (" (the others as their headers alone; '' writes the headers alone: the file's plan)"
                                if name == "convert" else ""))
    convert = commands.choices["convert"]
    convert.add_argument("--split-max-size", type=_size, default=0)
    convert.add_argument("--upstream-f32", action="store_true",
                         help="upstream's own conversion with every tensor F32, for upstream llama.cpp to run")
    # check reads the file's shards as they are: it writes nothing, so neither option applies
    commands.choices["check"].set_defaults(split_max_size=0, upstream_f32=False)
    commands.add_parser("verify-publish").add_argument("gguf", type=Path)
    logits = commands.add_parser("llama-logits")
    for name in ("gguf", "tokens", "logits"):
        logits.add_argument(name, type=Path)
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

    if args.command == "llama-logits":
        np.save(args.logits, llama.logits(args.gguf, np.load(args.tokens)))
        return

    if args.layers is not None and args.split_max_size:
        parser.error("a slice (--layers) is written in one file")
    url = f"https://huggingface.co/{args.source_repo}/tree/{args.source_revision}"
    model_type = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))["model_type"]
    common = {"source_repo_url": url, "layers": args.layers, "split_max_size": args.split_max_size,
              "shards": args.shards if args.command == "convert" else None}
    with torch.inference_mode():
        if model_type == "deepseek_v41":
            if args.upstream_f32:
                parser.error("--upstream-f32 is a GLM-5.3-Flash option")
            model = deepseek41.build(args.part, args.checkpoint, args.gguf, pool=args.pool, **common)
        elif model_type == "glm5_next":
            if args.pool is not None:
                parser.error("--pool is a DeepSeek V4.1 option")
            if args.upstream_f32:
                if args.part != "text" or args.split_max_size or args.shards is not None:
                    parser.error("--upstream-f32 writes the text model in one file")
                model = glm5next.upstream_f32(args.checkpoint, args.gguf, layers=args.layers)
            else:
                model = glm5next.build(args.part, args.checkpoint, args.gguf, **common)
        else:
            parser.error(f"no converter for model_type {model_type!r}")
        if args.command == "convert":
            model.write()
        else:
            checked = check.check(model, args.gguf, args.shards)
            logger.info("byte-exact: %s", ", ".join(f"{k} {v / 1e9:.3f} GB" for k, v in sorted(checked.items())))


if __name__ == "__main__":
    main()
