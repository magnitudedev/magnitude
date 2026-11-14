"""The csf_32x2_v1 decoder gives vLLM's bytes: on synthetic streams of every geometry the format has, and on every
routed scale of NVIDIA's GLM-5.3-Flash-FP4-Local when this host holds it; and it refuses what the streams can't state.

The oracle is vLLM's decoder, copied below as it stands at
https://github.com/vllm-project/vllm/blob/9b9bd049a73617e996d06cf35e6baf2a29ed03ae/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py
(Apache-2.0, Copyright contributors to the vLLM project)."""

from __future__ import annotations

import contextlib
import functools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from csf_streams import encode

from magnitude_conversion import csf


# --- vLLM, verbatim -------------------------------------------------------------------------------------------------
@functools.cache
def _csf32_blocks(rows: int, cols: int):
    """Origin, width and page of each 32x2 block in storage order, and the flat
    `[rows, cols]` position of each code."""
    tile_r, tile_c, pair, block = np.meshgrid(
        np.arange(-(-rows // 128)),
        np.arange(-(-cols // 4)),
        np.arange(2),
        np.arange(4),
        indexing="ij",
    )
    r0 = (128 * tile_r + 32 * block).ravel()
    c0 = (4 * tile_c + 2 * pair).ravel()
    page = (tile_r * tile_c.shape[1] + tile_c).ravel() // 32
    keep = (r0 < rows) & (c0 < cols)
    r0, c0, page = r0[keep], c0[keep], page[keep]
    width = np.minimum(cols - c0, 2)
    i, j = np.meshgrid(np.arange(32), np.arange(2), indexing="ij")
    cells = (r0[:, None, None] + i) * cols + c0[:, None, None] + j
    return r0, c0, width, page, cells[j < width[:, None, None]]


def _decode_csf_32x2(
    bases: torch.Tensor,
    codes: torch.Tensor,
    page_bases: torch.Tensor,
    offsets: torch.Tensor,
    exceptions: torch.Tensor,
) -> np.ndarray:
    dtypes = (torch.uint8, torch.uint8, torch.uint32, torch.uint16, torch.uint16)
    streams = (bases, codes, page_bases, offsets, exceptions)
    rows = bases.numel()
    cols = 2 * codes.numel() // max(rows, 1)
    if (
        any(t.dtype != d for t, d in zip(streams, dtypes))
        or rows % 32
        or rows * cols != 2 * codes.numel()
        or cols == 0
    ):
        raise ValueError("Invalid csf_32x2_v1 streams")
    r0, c0, width, page, cells = _csf32_blocks(rows, cols)
    num_pages = int(page[-1]) + 1
    base, packed, origins, index, words = (t.reshape(-1).numpy() for t in streams)
    if origins.size != num_pages or index.size != r0.size + num_pages:
        raise ValueError("Invalid csf_32x2_v1 index sizes")
    if (base > 240).any():
        raise ValueError("CSF row bases must be in 0..240")
    out = np.empty(rows * cols, dtype=np.uint8)
    out[cells] = np.stack((packed & 0xF, packed >> 4), -1).ravel() + base[cells // cols]

    first = np.flatnonzero(np.diff(page, prepend=-1))
    ends = origins[page].astype(np.int64) // 2 + index[np.arange(r0.size) + page + 1]
    starts = np.concatenate(([0], ends[:-1]))
    if (
        (index[first + np.arange(num_pages)] != 0).any()
        or (origins != 2 * starts[first]).any()
        or (ends < starts).any()
        or (ends - starts > 32 * width).any()
        or ends[-1] != words.size
    ):
        raise ValueError("Invalid csf_32x2_v1 exception index")
    block = np.repeat(np.arange(r0.size), ends - starts)
    pos = (words & 0xFF).astype(np.int64)
    if (
        (pos >= 64).any()
        or (pos % 2 >= width[block]).any()
        or ((pos[1:] <= pos[:-1]) & (block[1:] == block[:-1])).any()
    ):
        raise ValueError("CSF exception positions must be increasing and in range")
    out[(r0[block] + pos // 2) * cols + c0[block] + pos % 2] = words >> 8
    return out.reshape(rows, cols)
# --- end of vLLM ----------------------------------------------------------------------------------------------------


def _vllm(streams: dict[str, np.ndarray]) -> np.ndarray:
    return _decode_csf_32x2(*(torch.from_numpy(streams[s]) for s in csf.STREAMS))


# rows x cols: whole tiles; several pages (34 tiles); a partial last tile row; an odd last column (1-wide blocks)
GEOMETRIES = [(128, 4), (4096, 128), (256, 68), (160, 8), (96, 7), (32, 1)]


@pytest.mark.parametrize("rows, cols", GEOMETRIES)
@pytest.mark.parametrize("spread, below", [(16, 0), (64, 0), (256, 0), (16, 16)],
                         ids=["no-exceptions", "some-exceptions", "many-exceptions", "every-scale-an-exception"])
def test_decode_is_vllms(rows, cols, spread, below):
    """Each row's bytes lie within `spread` of a random offset; its base lies `below` under its smallest byte."""
    rng = np.random.default_rng(rows * cols + spread + below)
    scales = (rng.integers(0, spread, (rows, cols)) + rng.integers(below, 257 - spread, (rows, 1))).astype(np.uint8)
    streams = encode(scales, (np.minimum(scales.min(axis=1), 240) - below).astype(np.uint8))
    if below:
        assert streams["exceptions"].size == scales.size
    elif spread > 16 and cols > 1:
        assert streams["exceptions"].size
    decoded = csf.decode("w", rows, cols, streams)
    assert np.array_equal(decoded, scales)
    assert np.array_equal(decoded, _vllm(streams))


def _refused(rows: int, cols: int, change, match: str) -> None:
    scales = np.random.default_rng(0).integers(0, 256, (rows, cols)).astype(np.uint8)
    streams = encode(scales)
    change(streams)
    with pytest.raises(ValueError, match=match):
        csf.decode("w", rows, cols, streams)
    with pytest.raises(ValueError):  # not stricter than vLLM: it refuses the same streams
        _vllm(streams)


def _set(stream: str, index: int, value: int):
    def change(streams: dict[str, np.ndarray]) -> None:
        streams[stream] = streams[stream].copy()
        streams[stream][index] = value
    return change


def _resize(stream: str, size: int):
    def change(streams: dict[str, np.ndarray]) -> None:
        streams[stream] = streams[stream][:size]
    return change


@pytest.mark.parametrize("change, match", [
    (_set("bases", 3, 241), "base is 241, above 240"),
    (_set("exceptions", 0, 64 | 7 << 8), "outside its block or out of order"),
    (_set("exceptions", 1, 0), "outside its block or out of order"),
    (_set("offsets", 2, 0), "exception index disagrees"),
    (_set("page_bases", 1, 2), "exception index disagrees"),
    (_resize("exceptions", -1), "exception index disagrees"),
    (_resize("codes", -1), r"csf32_codes is uint8 \[\d+\]"),
], ids=["base-above-240", "position-out-of-range", "positions-out-of-order", "index-out-of-step",
        "page-base-out-of-step", "exceptions-short", "codes-short"])
def test_refusals(change, match):
    _refused(256, 68, change, match)


def test_an_exception_in_the_missing_column_of_a_one_wide_block_is_refused():
    scales = np.zeros((32, 1), dtype=np.uint8)
    streams = encode(scales)
    streams["exceptions"] = np.array([1 | 200 << 8], dtype=np.uint16)  # position 1: column 1 of a 1-wide block
    streams["offsets"] = np.array([0, 1], dtype=np.uint16)
    with pytest.raises(ValueError, match="outside its block"):
        csf.decode("w", 32, 1, streams)


def test_streams_of_another_shape_are_refused():
    streams = encode(np.zeros((128, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"csf32_bases is uint8 \[128\], a \[256, 4\] scale's is uint8 \[256\]"):
        csf.decode("w", 256, 4, streams)


# NVIDIA's QAD checkpoint, when this host holds it: every routed expert's scales
_CHECKPOINT = Path(os.environ.get("MAGNITUDE_MODELS", "/nonexistent")) / "nvidia/GLM-5.3-Flash-FP4-Local"


def _compare(work: list[tuple[str, int, int, dict[str, Path]]]) -> int:
    """Compares each scale's decoding with vLLM's, its streams read from their files."""
    with contextlib.ExitStack() as stack:
        files = {}
        for scale, rows, cols, parts in work:
            streams = {}
            for s, path in parts.items():
                if path not in files:
                    files[path] = stack.enter_context(safe_open(path, framework="numpy"))
                streams[s] = files[path].get_tensor(csf.stream_name(scale, s))
            if not np.array_equal(csf.decode(scale, rows, cols, streams), _vllm(streams)):
                raise AssertionError(f"{scale}: our bytes differ from vLLM's")
    return len(work)


@pytest.mark.skipif(not _CHECKPOINT.is_dir(), reason="NVIDIA's GLM-5.3-Flash-FP4-Local is not on this host")
def test_every_routed_scale_of_the_release_decodes_to_vllms_bytes():
    weight_map = json.loads((_CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    tc = json.loads((_CHECKPOINT / "config.json").read_text())["text_config"]
    scales = [name.removesuffix(".csf32_bases") for name in weight_map if name.endswith(".weight_scale.csf32_bases")]
    weights: dict[str, tuple[int, int]] = {}
    for file in {weight_map[scale.removesuffix("_scale")] for scale in scales}:
        with safe_open(_CHECKPOINT / file, framework="numpy") as f:
            weights |= {name: tuple(f.get_slice(name).get_shape()) for name in f.keys() if name.endswith(".weight")}
    by_file: dict[str, list[tuple[str, int, int, dict[str, Path]]]] = {}
    for scale in scales:
        rows, half = weights[scale.removesuffix("_scale")]
        parts = {s: _CHECKPOINT / weight_map[csf.stream_name(scale, s)] for s in csf.STREAMS}
        by_file.setdefault(weight_map[csf.stream_name(scale, "bases")], []).append((scale, rows, half * 2 // 16, parts))
    with ProcessPoolExecutor(min(len(by_file), os.cpu_count() or 1)) as pool:
        compared = sum(pool.map(_compare, [work for _, work in sorted(by_file.items())]))
    print(f"{compared} routed scales decode to vLLM's bytes")
    assert compared == tc["n_routed_experts"] * 3 * (tc["num_hidden_layers"] - tc["first_k_dense_replace"])
