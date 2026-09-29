"""Split-context decode attention reading the physical KV slab directly."""

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.kernels.attention import plans

from ..state.decode import DecodeKV
from ..state.views import PagedKV
from .contracts import PagedAttention
from .gathered import GatheredAttention, validate_attention


@component("MODEL:ATTENTION:MAG:PAGED")
class MetalPagedAttention:
    """Short query blocks use page addresses; prefill delegates to its own operator.

    Each SIMD group produces an FP32 online-softmax partial over a bounded key
    interval. A second pass combines those partials without gathering history or
    materializing a query-by-context score matrix. Physical state owns append,
    copy-on-write and execution pins; this operator owns only attention execution.
    """

    partition_tokens = 128

    def __init__(self, prefill: PagedAttention | None = None, *, heads_per_group: int = 2):
        if heads_per_group not in (1, 2, 4):
            raise ValueError("native attention head sharing must be 1, 2 or 4")
        self.prefill = prefill if prefill is not None else GatheredAttention()
        self.heads_per_group = heads_per_group

    @staticmethod
    def supports(queries: mx.array, kv: PagedKV) -> bool:
        return (
            1 <= queries.shape[2] <= 8
            and kv.keys.shape[-1] in (32, 64, 128, 256, 512)
            and kv.values.shape[-1] in (32, 64, 128, 256, 512)
            and queries.dtype == kv.keys.dtype == kv.values.dtype
            and queries.dtype in (mx.float32, mx.float16, mx.bfloat16)
        )

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
        if key_ends is not None:
            return self.prefill.compute(queries, kv, scale, window=window, key_ends=key_ends)
        if not self.supports(queries, kv):
            return self.prefill.compute(queries, kv, scale, window=window)
        count = queries.shape[2]
        covered = max(kv.lengths) if window is None else min(max(kv.lengths), window + count - 1)
        return self.apply(
            queries,
            kv.keys,
            kv.values,
            kv.table.device,
            mx.array([length - count for length in kv.lengths], mx.int32),
            page_size=kv.page_size,
            table_width=kv.table.width,
            covered=covered,
            scale=scale,
            window=window,
            tail=kv.tail_batch(),
        )

    def apply(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        pages: mx.array,
        positions: mx.array,
        *,
        page_size: int,
        table_width: int,
        covered: int,
        scale: float,
        window: int | None = None,
        tail: tuple[mx.array, mx.array, mx.array] | None = None,
    ) -> mx.array:
        return plans.attend(
            queries,
            keys,
            values,
            pages,
            positions,
            page_size=page_size,
            table_width=table_width,
            covered=covered,
            scale=scale,
            window=window,
            tail=tail,
            partition_tokens=self.partition_tokens,
            heads_per_group=self.heads_per_group,
        )

    def decode(
        self,
        queries: mx.array,
        kv: DecodeKV,
        layer: int,
        scale: float,
        *,
        window: int | None = None,
    ) -> mx.array:
        """Bind the same attention operation to a prepared resident state transition."""
        covered = kv.page_size * kv.table_width
        return self.apply(
            queries,
            kv.keys[layer],
            kv.values[layer],
            kv.pages,
            kv.positions,
            page_size=kv.page_size,
            table_width=kv.table_width,
            covered=min(covered, window) if window else covered,
            scale=scale,
            window=window,
            tail=kv.tail(layer),
        )
