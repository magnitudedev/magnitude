"""Attention reads model-owned staged KV; it never acquires append authority."""

import mlx.core as mx

from magnitude_engine.components import component

from ..state.views import PagedKV


@component("MODEL:ATTENTION:MAG:GATHERED")
class GatheredAttention:
    """Reference execution over logical KV, including heterogeneous causal windows."""

    def compute(
        self,
        queries: mx.array,
        kv: PagedKV,
        scale: float,
        *,
        window: int | None = None,
        key_ends: mx.array | None = None,
    ) -> mx.array:
        validate_attention(queries, kv, window, key_ends)
        count = queries.shape[2]
        starts = tuple(
            0 if window is None else max(0, length - count + 1 - window) for length in kv.lengths
        )
        longest = max(length - start for length, start in zip(kv.lengths, starts, strict=True))
        histories = []
        for row, (length, start) in enumerate(zip(kv.lengths, starts, strict=True)):
            k, v = kv.gather(row, start)
            padding = longest - (length - start)
            histories.append(
                (
                    mx.pad(k, [(0, 0), (0, padding), (0, 0)]),
                    mx.pad(v, [(0, 0), (0, padding), (0, 0)]),
                )
            )
        positions = mx.array([length - count for length in kv.lengths], mx.int32)
        queries_at = positions[:, None, None] + mx.arange(count)[None, :, None]
        keys_at = mx.array(starts, mx.int32)[:, None, None] + mx.arange(longest)[None, None, :]
        mask = (
            keys_at <= queries_at
            if key_ends is None
            else (
                (keys_at < key_ends[:, :, None])
                & (keys_at < mx.array(kv.lengths, mx.int32)[:, None, None])
            )
        )
        if window is not None:
            mask = mask & (keys_at > queries_at - window)
        return mx.fast.scaled_dot_product_attention(
            queries,
            mx.stack([k for k, _ in histories]),
            mx.stack([v for _, v in histories]),
            scale=scale,
            mask=mask[:, None],
        )


def validate_attention(
    queries: mx.array,
    kv: PagedKV,
    window: int | None,
    key_ends: mx.array | None = None,
) -> None:
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError("attention window must be a positive integer")
    if (
        queries.ndim != 4
        or queries.shape[0] != len(kv.lengths)
        or queries.shape[2] < 1
        or min(kv.lengths) < queries.shape[2]
        or queries.shape[1] < 1
        or queries.shape[1] % kv.keys.shape[0]
        or queries.shape[-1] != kv.keys.shape[-1]
    ):
        raise ValueError("attention queries do not match visible KV geometry")
    if key_ends is not None and (
        key_ends.dtype != mx.int32 or key_ends.shape != (queries.shape[0], queries.shape[2])
    ):
        raise ValueError("attention key ends must be exclusive [rows, queries] int32 bounds")
