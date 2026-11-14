"""ModelOpt's `csf_32x2_v1` compression of an NVFP4 weight's E4M3 block scales, decoded to the exact scale bytes.

A scale matrix [rows, cols] (one byte per 16 weights along a row) is stored as five streams, `<scale>.csf32_<stream>`:

- `bases` (U8 [rows]): each row's base byte, at most 240;
- `codes` (U8 [rows * cols / 2]): one 4-bit code per scale, two per byte, low nibble first, in storage order: 128x4
  tiles row-major, within a tile its two column pairs, within a pair its four 32-row blocks, within a block its
  32 rows x 2 columns row-major; a scale is its row's base plus its code;
- `exceptions` (U16): the scales a code doesn't state, block by block in storage order, each its in-block position
  (row * 2 + column) in the low byte and its scale byte in the high one, positions increasing within a block;
- `page_bases` (U32, one per page of 32 tiles): the byte offset in `exceptions` of the page's first exception;
- `offsets` (U16): per page a 0, then each of its blocks' exception end, counted from the page's first exception.

A last tile or block may be partial: a block of the last odd column is one column wide. Rows come in whole blocks.

The decoder is written from vLLM's `_decode_csf_32x2` (`vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`)
and compared with it byte for byte (`tests/test_csf.py`). It takes the matrix's shape from its weight rather than from
the streams, so streams of another size are refused, as is anything they can't state: a base above 240, an index
that disagrees with the exceptions, an exception outside its block or out of order.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import numpy as np

STREAMS = ("bases", "codes", "page_bases", "offsets", "exceptions")
_DTYPES = {"bases": np.uint8, "codes": np.uint8, "page_bases": np.uint32, "offsets": np.uint16,
           "exceptions": np.uint16}
_MAX_BASE = 240  # base + a 4-bit code stays a byte


def stream_name(scale: str, stream: str) -> str:
    return f"{scale}.csf32_{stream}"


@functools.cache
def _blocks(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Each 32x2 block in storage order: its first row and column, width and page; the page count; and the flat
    `[rows, cols]` position of each code, in code order."""
    tiles_c = -(-cols // 4)
    tile_r, tile_c, pair, block = np.meshgrid(np.arange(-(-rows // 128)), np.arange(tiles_c), np.arange(2),
                                              np.arange(4), indexing="ij")
    r0, c0 = (128 * tile_r + 32 * block).ravel(), (4 * tile_c + 2 * pair).ravel()
    tile = (tile_r * tiles_c + tile_c).ravel()
    kept = (r0 < rows) & (c0 < cols)
    r0, c0, page = r0[kept], c0[kept], tile[kept] // 32
    width = np.minimum(cols - c0, 2)
    i, j = np.meshgrid(np.arange(32), np.arange(2), indexing="ij")
    cells = ((r0[:, None, None] + i) * cols + c0[:, None, None] + j)[j < width[:, None, None]]
    return r0, c0, width, page, int(page[-1]) + 1, cells


def check_layout(name: str, rows: int, cols: int, layouts: Mapping[str, tuple[np.dtype, tuple[int, ...]]]) -> None:
    """Raises naming scale `name` unless streams of these (type, shape)s can hold a [rows, cols] scale."""
    if rows <= 0 or cols <= 0 or rows % 32:
        raise ValueError(f"{name}: a csf_32x2_v1 scale has whole 32-row blocks, not [{rows}, {cols}]")
    r0, _, _, _, pages, _ = _blocks(rows, cols)
    sizes = {"bases": rows, "codes": rows * cols // 2, "page_bases": pages, "offsets": r0.size + pages}
    for stream in STREAMS:
        dtype, shape = layouts[stream]
        if dtype != _DTYPES[stream] or len(shape) != 1 or shape[0] != sizes.get(stream, shape[0]):
            raise ValueError(f"{name}: csf32_{stream} is {dtype} {list(shape)}, a [{rows}, {cols}] scale's is "
                             f"{np.dtype(_DTYPES[stream])} [{sizes.get(stream, 'n')}]")


def decode(name: str, rows: int, cols: int, streams: Mapping[str, np.ndarray]) -> np.ndarray:
    """The scale bytes [rows, cols] (U8) of the streams of scale `name`; raises naming it on anything the streams
    can't state exactly."""
    check_layout(name, rows, cols, {s: (a.dtype, a.shape) for s, a in streams.items()})
    r0, c0, width, page, pages, cells = _blocks(rows, cols)
    bases, codes, page_bases, offsets, words = (streams[s] for s in STREAMS)
    if (bases > _MAX_BASE).any():
        raise ValueError(f"{name}: a row base is {int(bases.max())}, above {_MAX_BASE}")
    out = np.empty(rows * cols, dtype=np.uint8)
    out[cells] = np.stack((codes & 0xF, codes >> 4), -1).ravel() + bases[cells // cols]

    # each block's exceptions are words[start:end], ends counted from its page's first exception
    firsts = np.flatnonzero(np.diff(page, prepend=-1))
    ends = page_bases[page].astype(np.int64) // 2 + offsets[np.arange(r0.size) + page + 1]
    starts = np.concatenate(([0], ends[:-1]))
    if ((offsets[firsts + np.arange(pages)] != 0).any() or (page_bases != 2 * starts[firsts]).any()
            or (ends < starts).any() or (ends - starts > 32 * width).any() or ends[-1] != words.size):
        raise ValueError(f"{name}: the csf32 exception index disagrees with its {words.size} exceptions")
    block = np.repeat(np.arange(r0.size), ends - starts)
    position = (words & 0xFF).astype(np.int64)
    same_block = block[1:] == block[:-1]
    if ((position >= 64).any() or (position % 2 >= width[block]).any()
            or (same_block & (position[1:] <= position[:-1])).any()):
        raise ValueError(f"{name}: a csf32 exception lies outside its block or out of order")
    out[(r0[block] + position // 2) * cols + c0[block] + position % 2] = words >> 8
    return out.reshape(rows, cols)
