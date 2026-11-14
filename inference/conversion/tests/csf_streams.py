"""csf_32x2_v1 streams of given scale bytes, as ModelOpt compresses them, for tests and test checkpoints."""

from __future__ import annotations

import numpy as np

from magnitude_conversion import csf


def encode(scales: np.ndarray, bases: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """csf_32x2_v1 streams of the scale bytes [rows, cols] over the row bases given, by default each row's smallest
    byte (at most 240); a byte not within 15 above its base is an exception."""
    rows, cols = scales.shape
    if bases is None:
        bases = np.minimum(scales.min(axis=1), 240).astype(np.uint8)
    r0, c0, width, page, pages, cells = csf._blocks(rows, cols)
    flat = scales.reshape(-1)
    diff = flat[cells].astype(np.int64) - bases[cells // cols]
    nibbles = np.where((diff >= 0) & (diff <= 15), diff, 0).astype(np.uint8)
    codes = nibbles[0::2] | (nibbles[1::2] << 4)
    words, ends = [], []
    for b in range(r0.size):
        i, j = np.meshgrid(np.arange(32), np.arange(width[b]), indexing="ij")
        block = scales[r0[b]:r0[b] + 32, c0[b]:c0[b] + width[b]]
        diff = block.astype(np.int64) - bases[r0[b]:r0[b] + 32, None]
        far = (diff < 0) | (diff > 15)
        words += [(int(p) | int(v) << 8) for p, v in sorted(zip((i * 2 + j)[far], block[far]))]
        ends.append(len(words))
    offsets, page_bases = [], []
    for p in range(pages):
        in_page = np.flatnonzero(page == p)
        start = ends[in_page[0] - 1] if in_page[0] else 0
        page_bases.append(2 * start)
        offsets += [0] + [ends[b] - start for b in in_page]
    return {"bases": bases, "codes": codes.astype(np.uint8), "page_bases": np.array(page_bases, dtype=np.uint32),
            "offsets": np.array(offsets, dtype=np.uint16), "exceptions": np.array(words, dtype=np.uint16)}
