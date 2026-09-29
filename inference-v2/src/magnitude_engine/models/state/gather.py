"""Logical range reads from immutable heads-major page buffers."""

import mlx.core as mx

from .placement import runs


def gather_pages(
    keys: mx.array, values: mx.array, page_size: int, pages: tuple[int, ...], start: int, stop: int
) -> tuple[mx.array, mx.array]:
    if not 0 <= start <= stop <= len(pages) * page_size:
        raise ValueError("KV read range exceeds its page capacity")
    if start == stop:
        return (
            mx.zeros((keys.shape[0], 0, keys.shape[-1]), keys.dtype),
            mx.zeros((values.shape[0], 0, values.shape[-1]), values.dtype),
        )
    first, last = start // page_size, (stop + page_size - 1) // page_size
    selected = runs(pages[first:last])
    result = []
    for source in (keys, values):
        pieces = [source[:, run.start * page_size : run.end * page_size] for run in selected]
        joined = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)
        offset = start - first * page_size
        result.append(mx.contiguous(joined[:, offset : offset + stop - start]))
    return result[0], result[1]
