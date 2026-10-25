#!/usr/bin/env python3
"""Reference dequantization of GGUF tensor types, from gguf-py (`gguf.quants`,
the version `inference/evaluation/references` pins), for the engine's import
tests. Q1_0, which gguf-py does not dequantize, is a port of ggml-quants.c
`dequantize_row_q1_0`.

`vectors DIR` writes one `<TYPE>.bin` per weight type: seeded random blocks
of the codes a GGUF file can hold whose reference values are all finite,
then those values. The file is the
little-endian u32 block count, the blocks' bytes, and their f32 values.
`engine/kernels/tests/import_residency.rs` checks the registry's conversion
and decode of every type against them.

`dump GGUF DIR` writes real tensors of a GGUF file (their first `--rows`
rows) with their reference values, for the ignored real-tensor test.

Run with `uv run --project inference/evaluation/references python -I` and
this file's path."""

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T
from gguf.quants import dequantize

BLOCKS = 16
# Types that are not weight formats: activation-side blocks no producer
# writes as tensors, and the dense and integer types.
NOT_WEIGHTS = {T.Q8_1, T.Q8_K, T.F64, T.I8, T.I16, T.I32, T.I64, T.F32, T.F16, T.BF16}


def dequantize_q1_0(data):
    # ggml-quants.c dequantize_row_q1_0: d (f16), then 128 bits, value j is
    # d when bit j is set, else -d.
    blocks = data.reshape(-1, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    bits = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little").astype(bool)
    return np.where(bits, d, -d).reshape(-1)


def reference(data, qtype):
    # Random blocks may hold NaN or infinite scales; `vectors` draws again.
    with np.errstate(invalid="ignore", over="ignore"):
        if qtype == T.Q1_0:
            return dequantize_q1_0(data)
        return dequantize(data, qtype).astype(np.float32).reshape(-1)


def vectors(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for qtype in T:
        if qtype in NOT_WEIGHTS:
            continue
        values_per_block, bytes_per_block = GGML_QUANT_SIZES[qtype]
        rng = np.random.default_rng(zlib.crc32(qtype.name.encode()))
        blocks, values = [], []
        while len(blocks) < BLOCKS:
            block = rng.integers(0, 256, bytes_per_block, dtype=np.uint8)
            if qtype == T.NVFP4:
                # Its four UE4M3 scales over the codes ggml's quantizer writes
                # (`ggml_fp32_to_ue4m3`: 0x00 ..= 0x7e). The engine decodes
                # UE4M3 as NVFP4 hardware does (0x7f NaN, sign ignored), where
                # ggml decodes 0x7f as 0 and 0xff as 240, which no file holds.
                block[:4] = rng.integers(0, 0x7F, 4, dtype=np.uint8)
            decoded = reference(block, qtype)
            assert decoded.shape == (values_per_block,), qtype.name
            if np.isfinite(decoded).all():
                blocks.append(block)
                values.append(decoded)
        with open(directory / f"{qtype.name}.bin", "wb") as out:
            out.write(struct.pack("<I", BLOCKS))
            out.write(np.concatenate(blocks).tobytes())
            out.write(np.concatenate(values).astype("<f4").tobytes())


def dump(gguf_path, directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for tensor in GGUFReader(gguf_path).tensors:
        qtype = T(tensor.tensor_type)
        if qtype in NOT_WEIGHTS or len(tensor.shape) != 2:
            continue
        k = int(tensor.shape[0])
        values_per_block, bytes_per_block = GGML_QUANT_SIZES[qtype]
        row_bytes = k // values_per_block * bytes_per_block
        taken = min(rows, int(tensor.shape[1]))
        data = np.asarray(tensor.data).reshape(-1)[: taken * row_bytes].view(np.uint8)
        stem = tensor.name.replace("/", "_")
        (directory / f"{stem}.blocks").write_bytes(data.tobytes())
        (directory / f"{stem}.values").write_bytes(reference(data, qtype).astype("<f4").tobytes())
        index.append(f"{qtype.name}\t{taken}\t{k}\t{stem}")
    (directory / "index.tsv").write_text("\n".join(index) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("vectors").add_argument("directory", type=Path)
    dumping = commands.add_parser("dump")
    dumping.add_argument("gguf", type=Path)
    dumping.add_argument("directory", type=Path)
    dumping.add_argument("--rows", type=int, default=64)
    arguments = parser.parse_args()
    if arguments.command == "vectors":
        vectors(arguments.directory)
    else:
        dump(arguments.gguf, arguments.directory, arguments.rows)


if __name__ == "__main__":
    main()
