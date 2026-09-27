#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0"]
# ///
"""Truncated real-weight GGUFs for models too large to run whole (plan §3.9).

uv run inference/validation/layer_slice.py --repository ORG/NAME --revision SHA --path FIRST-SHARD.gguf \\
    --output slice.gguf [--layers 0,1,4] [--minimum 4] [--no-mtp] [--list]
uv run inference/validation/layer_slice.py --model LOCAL.gguf --output slice.gguf ...

Reads the header of every shard of a locked Hugging Face revision by HTTP byte range and fetches
only the byte ranges of the tensors it keeps: the global tensors (embedding, final norm, head, rope
tables, ...), the selected main layers renumbered 0..n-1, and the trailing MTP (`nextn`) layers
renumbered after them. Metadata is copied unchanged except `block_count`, per-layer arrays (entries
of the kept layers) and the dropped `split.*` keys, so the slice is recognized as the same family.

Default selection: the shortest prefix of the main layers that contains every layer *variant* and at
least `--minimum` layers. A variant is a layer's tensor names and shapes plus its entries in every
per-layer metadata array; tensor types are not part of it. A prefix keeps implicit periodic layer
patterns (a sliding-window period, a full-attention interval, a dense-first layer) intact, which
`--minimum 4` covers for the in-scope families. `--layers` picks an explicit ascending list instead;
the caller then owns the pattern keys. `--list` prints the variants and the default selection.

A provenance record `<output>.json` lists the source files, kept layers, renaming and output sha256.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
import json
from pathlib import Path
import re

from gguf.constants import GGUFValueType

from reference_gguf import HEADER_CACHE, Package, Value, Writer

LAYER = re.compile(r"^blk\.(\d+)\.(.+)$")
# Metadata counting trailing main layers (Gemma 4's KV-shared layers). A slice keeps a trailing
# layer's role only when every kept layer after it is also trailing, which a prefix-plus-tail
# selection guarantees and `--layers` must respect.
TRAILING_COUNTS = ("attention.shared_kv_layers",)
# Global tensors that pack one slice per layer (Gemma 4 per-layer embeddings); a slice would have to
# cut them too, so such models are only sliced with every main layer kept (they are small anyway).
LAYER_PACKED = ("per_layer_token_embd.weight", "per_layer_model_proj.weight")


def split_keys(package: Package) -> dict[str, Value]:
    return {key: value for key, value in package.metadata.items() if not key.startswith("split.")}


class Slicer:
    def __init__(self, package: Package):
        self.package = package
        self.architecture = package.architecture
        self.total = package.key("block_count")
        self.mtp = package.key("nextn_predict_layers", 0)
        self.main = self.total - self.mtp
        self.per_layer_keys = [key for key, value in package.metadata.items()
                               if value.kind == GGUFValueType.ARRAY and len(value.value) == self.total
                               and key.startswith(f"{self.architecture}.")]
        self.layer_tensors: dict[int, list[str]] = {}
        self.global_tensors: list[str] = []
        for name in package.names():
            match = LAYER.match(name)
            if match:
                self.layer_tensors.setdefault(int(match[1]), []).append(name)
            else:
                self.global_tensors.append(name)

    def variant(self, layer: int) -> tuple:
        tensors = tuple(sorted((LAYER.match(name)[2], self.package.info(name).shape) for name in self.layer_tensors[layer]))
        metadata = tuple((key, json.dumps(self.package.metadata[key].value[layer])) for key in self.per_layer_keys)
        return tensors, metadata

    def units(self, layer: int) -> set:
        """Coverage units a layer completes: its variant and the variant pair it forms with the previous
        layer (a sublayer family such as Nemotron-H pairs consecutive layers into blocks)."""
        units = {self.variant(layer)}
        if layer > 0:
            units.add((self.variant(layer - 1), self.variant(layer)))
        return units

    def default_layers(self, minimum: int) -> list[int]:
        wanted = set().union(*(self.units(layer) for layer in range(self.main)))
        seen = set()
        for layer in range(self.main):
            seen |= self.units(layer)
            if seen == wanted and layer + 1 >= minimum:
                return list(range(layer + 1))
        return list(range(self.main))

    def metadata(self, kept: list[int]) -> dict[str, Value]:
        metadata = split_keys(self.package)
        key = f"{self.architecture}.block_count"
        metadata[key] = Value(metadata[key].kind, len(kept))
        for key in self.per_layer_keys:
            value = metadata[key]
            metadata[key] = Value(value.kind, [value.value[layer] for layer in kept], value.item)
        # Counts of trailing layers keep meaning "the last n layers": recount over the kept layers.
        for suffix in TRAILING_COUNTS:
            key = f"{self.architecture}.{suffix}"
            if key in metadata:
                first = self.main - metadata[key].value
                metadata[key] = Value(metadata[key].kind, sum(first <= layer < self.main for layer in kept))
        return metadata

    def tensors(self, kept: list[int]) -> list[tuple[str, str]]:
        """(source name, output name) in output order: globals first, then layers in the new order."""
        pairs = [(name, name) for name in self.global_tensors]
        for new, old in enumerate(kept):
            pairs += [(name, f"blk.{new}.{LAYER.match(name)[2]}") for name in self.layer_tensors[old]]
        return pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repository")
    source.add_argument("--model", type=Path, help="a local (possibly split) GGUF")
    parser.add_argument("--revision")
    parser.add_argument("--path", help="first shard path inside the repository")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layers", help="comma-separated ascending main-layer indices")
    parser.add_argument("--minimum", type=int, default=4, help="minimum main layers in the default prefix")
    parser.add_argument("--no-mtp", action="store_true", help="drop the trailing MTP layers")
    parser.add_argument("--list", action="store_true", help="print variants and the default selection only")
    parser.add_argument("--parallel", type=int, default=8, help="concurrent tensor range reads")
    options = parser.parse_args()
    if options.repository:
        if not (options.revision and options.path):
            parser.error("--repository needs --revision and --path")
        package = Package.remote(options.repository, options.revision, options.path, HEADER_CACHE)
    else:
        package = Package.local(options.model)
    slicer = Slicer(package)
    main_layers = [int(x) for x in options.layers.split(",")] if options.layers else slicer.default_layers(options.minimum)
    if main_layers != sorted(set(main_layers)) or main_layers[-1] >= slicer.main:
        parser.error(f"--layers must be ascending main-layer indices below {slicer.main}")
    if any(package.has(name) for name in LAYER_PACKED) and len(main_layers) != slicer.main:
        parser.error(f"{', '.join(n for n in LAYER_PACKED if package.has(n))} pack every layer; keep all main layers")
    variants: dict[tuple, list[int]] = {}
    for layer in range(slicer.main):
        variants.setdefault(slicer.variant(layer), []).append(layer)
    if options.list:
        print(json.dumps({"architecture": slicer.architecture, "main_layers": slicer.main, "mtp_layers": slicer.mtp,
                          "per_layer_keys": slicer.per_layer_keys,
                          "variants": [{"layers": layers, "tensors": [t[0] for t in variant[0]]} for variant, layers in variants.items()],
                          "default_layers": main_layers}, indent=1))
        return
    if options.output is None:
        parser.error("--output is required unless --list")
    missing = [layers for layers in variants.values() if not set(layers) & set(main_layers)]
    if missing:
        print(f"note: layer variants not in the selection (first layers): {[layers[0] for layers in missing]}")
    mtp = [] if options.no_mtp else list(range(slicer.main, slicer.total))
    kept = main_layers + mtp
    metadata = slicer.metadata(kept)
    if options.no_mtp and slicer.mtp:
        key = f"{slicer.architecture}.nextn_predict_layers"
        metadata[key] = Value(metadata[key].kind, 0)
    pairs = slicer.tensors(kept)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    writer = Writer(options.output, metadata)
    for old, new in pairs:
        info = package.info(old)
        writer.declare(new, info.shape, info.type)
    writer.begin()
    total_bytes = 0
    # At most `parallel` tensors are in flight or waiting to be written, so memory stays bounded.
    with ThreadPoolExecutor(options.parallel) as pool:
        pending: deque = deque()
        queue = iter(pairs)
        for pair in islice(queue, options.parallel):
            pending.append((pair, pool.submit(lambda name: bytes(package.raw(name)), pair[0])))
        while pending:
            (old, new), future = pending.popleft()
            data = future.result()
            writer.write(new, data)
            total_bytes += len(data)
            print(f"{old} -> {new} ({len(data) / 2**20:.1f} MiB)", flush=True)
            for pair in islice(queue, 1):
                pending.append((pair, pool.submit(lambda name: bytes(package.raw(name)), pair[0])))
    writer.close()
    record = {
        "source": [source.describe() for source in package.sources], "architecture": slicer.architecture,
        "main_layers": main_layers, "mtp_layers": mtp, "renaming": {str(old): new for new, old in enumerate(kept)},
        "tensor_bytes": total_bytes, "output": str(options.output), "sha256": sha256_file(options.output),
    }
    options.output.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "renaming"}, indent=1))


if __name__ == "__main__":
    main()
