"""Split-context decode attention reading the physical KV slab directly."""

from functools import cache
from typing import Any

import mlx.core as mx

from ..state.views import PagedKV
from .contracts import PagedAttention
from .gathered import GatheredAttention, validate_attention


@cache
def _partials() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_paged_attention_partials",
        input_names=["queries", "keys", "values", "pages", "positions", "layout", "scale"],
        output_names=["partial"],
        source="""
        uint lane = thread_position_in_grid.x;
        uint split = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;
        const uint splits = layout[2];
        if (split >= splits) return;
        uint sequence = row / ((HQ / HEADS) * TQ);
        uint head = ((row / TQ) % (HQ / HEADS)) * HEADS;
        uint token = row % TQ;
        uint kh = head / (HQ / HK);
        uint visible = positions[sequence] + token + 1;
        uint partition_base = 0;
        if (layout[4] > 0 && positions[sequence] + 1 > uint(layout[4]))
            partition_base = positions[sequence] + 1 - uint(layout[4]);
        uint first = partition_base + split * SPAN;
        if (layout[4] > 0 && visible > uint(layout[4]))
            first = max(first, visible - uint(layout[4]));
        uint last = min(partition_base + (split + 1) * SPAN, visible);
        float query[HEADS][DK / 32];
        float accumulator[HEADS][DV / 32];
        float maximum[HEADS];
        float denominator[HEADS];
        for (uint h = 0; h < HEADS; ++h) {
            for (uint i = 0; i < DK / 32; ++i)
                query[h][i] = float(queries[
                    ((sequence * HQ + head + h) * TQ + token) * DK + lane + 32 * i]);
            for (uint i = 0; i < DV / 32; ++i) accumulator[h][i] = 0.0f;
            maximum[h] = -INFINITY;
            denominator[h] = 0.0f;
        }
        uint position = first;
        while (position < last) {
            // Resolve a physical page once, then consume its contiguous keys.
            uint page_index = position / layout[1];
            uint page_end = min(last, (page_index + 1) * layout[1]);
            uint physical = pages[sequence * layout[3] + page_index] * layout[1]
                + position % layout[1];
            size_t address = size_t(kh) * layout[0] + physical;
            for (; position < page_end; ++position, ++address) {
                float key[DK / 32];
                float value[DV / 32];
                for (uint i = 0; i < DK / 32; ++i)
                    key[i] = float(keys[address * DK + lane + 32 * i]);
                for (uint i = 0; i < DV / 32; ++i)
                    value[i] = float(values[address * DV + lane + 32 * i]);
                for (uint h = 0; h < HEADS; ++h) {
                    float score = 0.0f;
                    for (uint i = 0; i < DK / 32; ++i) score += query[h][i] * key[i];
                    score = simd_sum(score) * scale[0];
                    float next_maximum = max(maximum[h], score);
                    float previous_weight = exp(maximum[h] - next_maximum);
                    float weight = exp(score - next_maximum);
                    denominator[h] = denominator[h] * previous_weight + weight;
                    for (uint i = 0; i < DV / 32; ++i)
                        accumulator[h][i] = accumulator[h][i] * previous_weight + weight * value[i];
                    maximum[h] = next_maximum;
                }
            }
        }
        for (uint h = 0; h < HEADS; ++h) {
            size_t output_row = (sequence * HQ + head + h) * TQ + token;
            size_t destination = (output_row * splits + split) * (DV + 2);
            for (uint i = 0; i < DV / 32; ++i)
                partial[destination + lane + 32 * i] = accumulator[h][i];
            if (lane == 0) {
                partial[destination + DV] = maximum[h];
                partial[destination + DV + 1] = denominator[h];
            }
        }
        """,
    )


@cache
def _combine() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_paged_attention_combine",
        input_names=["partial", "layout"],
        output_names=["output"],
        source="""
        uint lane = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        const uint splits = layout[2];
        size_t base = size_t(row) * splits * (DV + 2);
        float maximum = -INFINITY;
        for (uint split = lane; split < splits; split += 32)
            maximum = max(maximum, partial[base + split * (DV + 2) + DV]);
        maximum = simd_max(maximum);
        float denominator = 0.0f;
        float result[DV / 32];
        for (uint i = 0; i < DV / 32; ++i) result[i] = 0.0f;
        for (uint split = 0; split < splits; ++split) {
            size_t address = base + split * (DV + 2);
            float weight = exp(partial[address + DV] - maximum);
            denominator += weight * partial[address + DV + 1];
            for (uint i = 0; i < DV / 32; ++i)
                result[i] += weight * partial[address + lane + 32 * i];
        }
        for (uint i = 0; i < DV / 32; ++i)
            output[row * DV + lane + 32 * i] = Out(result[i] / denominator);
        """,
    )


class MetalPagedAttention:
    """Short query blocks use page addresses; prefill delegates to its own operator.

    Each SIMD group produces an FP32 online-softmax partial over a bounded key
    interval. A second pass combines those partials without gathering history or
    materializing a query-by-context score matrix. Physical state owns append,
    copy-on-write and execution pins; this operator owns only attention execution.
    """

    partition_tokens = 128

    def __init__(self, prefill: PagedAttention | None = None, *, heads_per_group: int = 1):
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
        self, queries: mx.array, kv: PagedKV, scale: float, *, window: int | None = None
    ) -> mx.array:
        validate_attention(queries, kv, window)
        if not self.supports(queries, kv):
            return self.prefill.compute(queries, kv, scale, window=window)
        count = queries.shape[2]
        covered = max(kv.lengths) if window is None else min(max(kv.lengths), window + count - 1)
        return self.apply(
            queries, kv.keys, kv.values, kv.table.device,
            mx.array([length - count for length in kv.lengths], mx.int32),
            page_size=kv.page_size, table_width=kv.table.width, covered=covered,
            scale=scale, window=window,
        )

    def apply(
        self, queries: mx.array, keys: mx.array, values: mx.array,
        pages: mx.array, positions: mx.array, *,
        page_size: int, table_width: int, covered: int,
        scale: float, window: int | None = None,
    ) -> mx.array:
        """Pure launch on validated storage; positions and mappings remain tensor inputs."""
        count = queries.shape[2]
        span = self.partition_tokens
        splits = (covered + span - 1) // span
        rows = queries.shape[0] * queries.shape[1] * count
        dk, dv = keys.shape[-1], values.shape[-1]
        group = queries.shape[1] // keys.shape[0]
        heads = min(self.heads_per_group, group)
        while group % heads:
            heads -= 1
        threadgroup = (32, 4, 1)
        cells = (group // heads) * count
        if cells > 1:
            # Neighboring query/head cells share a KV head. Keep their SIMD groups
            # together, including short verification blocks, instead of separating
            # those identical history reads into different context partitions.
            sharing = min(4, cells)
            while cells % sharing:
                sharing -= 1
            threadgroup = (32, 1, sharing)
        layout = mx.array(
            [keys.shape[1], page_size, splits, table_width, window or 0], mx.int32
        )
        partial = _partials()(
            inputs=[
                queries,
                keys,
                values,
                pages,
                positions,
                layout,
                mx.array([scale], mx.float32),
            ],
            template=[
                ("DK", dk),
                ("DV", dv),
                ("HQ", queries.shape[1]),
                ("HK", keys.shape[0]),
                ("TQ", count),
                ("SPAN", span),
                ("HEADS", heads),
            ],
            grid=(32, splits, rows // heads),
            threadgroup=threadgroup,
            output_shapes=[(rows, splits, dv + 2)],
            output_dtypes=[mx.float32],
        )[0]
        return _combine()(
            inputs=[partial, layout],
            template=[("Out", queries.dtype), ("DV", dv)],
            grid=(32, rows, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(queries.shape[0], queries.shape[1], count, dv)],
            output_dtypes=[queries.dtype],
        )[0]
