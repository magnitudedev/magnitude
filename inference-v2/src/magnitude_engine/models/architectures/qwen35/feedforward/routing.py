"""Qwen routing epilogue: rounded softmax, stable top-k, normalization and shared gate."""

from functools import cache
from typing import Any

import mlx.core as mx


@cache
def _kernel() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_qwen_routes",
        input_names=["logits"],
        output_names=["indices", "scores", "shared"],
        source="""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint group = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.y;
        threadgroup float maxima[32];
        threadgroup float sums[32];
        threadgroup uint winners[32];
        threadgroup uint chosen;
        threadgroup float selected[TOPK];
        float values[4];
        float peak = -INFINITY;
        for (uint i = 0; i < 4; ++i) {
            uint index = tid * 4 + i;
            values[i] = index < EXPERTS ? float(logits[row * (EXPERTS + 1) + index]) : -INFINITY;
            peak = max(peak, values[i]);
        }
        if (group == 0) { maxima[lane] = -INFINITY; sums[lane] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        peak = simd_max(peak);
        if (lane == 0) maxima[group] = peak;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            peak = simd_max(maxima[lane]);
            if (lane == 0) maxima[0] = peak;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            values[i] = metal::fast::exp(values[i] - maxima[0]);
            total += values[i];
        }
        total = simd_sum(total);
        if (lane == 0) sums[group] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (group == 0) {
            total = simd_sum(sums[lane]);
            if (lane == 0) sums[0] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inverse = 1.0f / sums[0];
        for (uint i = 0; i < 4; ++i)
            values[i] = tid * 4 + i < EXPERTS ? float(T(values[i] * inverse)) : -1.0f;

        // MLX Metal argpartition uses a stable ascending sort. Select the largest
        // rounded probability/index pairs, and emit them in ascending order.
        for (uint rank = 0; rank < TOPK; ++rank) {
            float best = -1.0f;
            uint index = 0;
            for (uint i = 0; i < 4; ++i) {
                if (values[i] >= best) { best = values[i]; index = tid * 4 + i; }
            }
            float maximum = simd_max(best);
            uint winner = simd_max(best == maximum ? index : 0u);
            if (lane == 0) { maxima[group] = maximum; winners[group] = winner; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float value = maxima[0]; uint candidate = winners[0];
                for (uint g = 1; g < GROUPS; ++g) {
                    if (maxima[g] > value || (maxima[g] == value && winners[g] > candidate)) {
                        value = maxima[g]; candidate = winners[g];
                    }
                }
                chosen = candidate;
                indices[row * TOPK + TOPK - 1 - rank] = candidate;
                selected[TOPK - 1 - rank] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 4; ++i) if (tid * 4 + i == chosen) values[i] = -1.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            float norm = 0.0f;
            for (uint rank = 0; rank < TOPK; ++rank) norm = float(T(norm + selected[rank]));
            norm = float(T(norm));
            for (uint rank = 0; rank < TOPK; ++rank)
                scores[row * TOPK + rank] = T(NORMALIZE ? selected[rank] / norm : selected[rank]);
            T gate = logits[row * (EXPERTS + 1) + EXPERTS];
            T e = T(metal::precise::exp(float(metal::abs(gate))));
            auto y = 1 / (1 + e);
            shared[row] = gate < 0 ? y : 1 - y;
        }
        """,
    )


def select(logits: mx.array, top_k: int, normalize: bool) -> tuple[mx.array, mx.array, mx.array]:
    experts = logits.shape[-1] - 1
    if not 1 <= top_k <= min(16, experts) or not 1 <= experts <= 1024:
        raise ValueError("routing epilogue requires up to 1024 experts and 16 selected routes")
    threads = max(32, ((experts + 127) // 128) * 32)
    rows = logits.size // logits.shape[-1]
    return tuple(
        _kernel()(
            inputs=[logits],
            template=[
                ("T", logits.dtype),
                ("EXPERTS", experts),
                ("TOPK", top_k),
                ("GROUPS", threads // 32),
                ("NORMALIZE", normalize),
            ],
            grid=(threads, rows, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[
                (*logits.shape[:-1], top_k),
                (*logits.shape[:-1], top_k),
                (*logits.shape[:-1], 1),
            ],
            output_dtypes=[mx.uint32, logits.dtype, logits.dtype],
        )
    )
