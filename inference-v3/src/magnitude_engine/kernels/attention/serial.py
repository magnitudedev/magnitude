"""Streaming host schedule with O(head width) local storage per worker."""

import math

import tilelang.language as T

from magnitude_engine.platform.execution import DType


def run_attention(
    rows: int,
    query_heads: int,
    kv_heads: int,
    width: int,
    capacity: int,
    *,
    segments: int = 1,
    segment_capacity: int | None = None,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    """Streaming CPU schedule with O(head width) local storage per worker."""
    segment_capacity = capacity if segment_capacity is None else segment_capacity
    if (
        min(rows, query_heads, kv_heads, width, capacity, segments, segment_capacity) <= 0
        or query_heads % kv_heads
        or segment_capacity > capacity
    ):
        raise ValueError("invalid CPU attention geometry")

    @T.prim_func
    def main(
        Q: T.Tensor((rows, query_heads, width), dtype.value),
        K: T.Tensor((capacity, kv_heads, width), dtype.value),
        V: T.Tensor((capacity, kv_heads, width), dtype.value),
        Positions: T.Tensor((rows,), "int32"),
        Run: T.Tensor((segments, 3), "int32"),
        Output: T.Tensor((segments, rows, query_heads, width), output_dtype.value),
        Statistics: T.Tensor((segments, rows, query_heads, 2), "float32"),
    ):
        for segment, row, head in T.Parallel(segments, rows, query_heads):
            physical = T.if_then_else(segments == 1, 0, Run[segment, 0])
            output = T.alloc_local((width,), "float32")
            maximum = T.alloc_local((1,), "float32")
            denominator = T.alloc_local((1,), "float32")
            score = T.alloc_local((1,), "float32")
            maximum[0] = -3.402823466e38
            denominator[0] = 0
            for d in T.serial(width):
                output[d] = 0
            # A static segment bound avoids the CPU tile-loop fuser's invalid
            # rewrite of worker-indexed dynamic bounds. Only visible rows load.
            for token in T.serial(segment_capacity):
                if token < Run[segment, 2] and Run[segment, 1] + token <= Positions[row]:
                    score[0] = 0
                    for d in T.serial(width):
                        score[0] += Q[row, head, d].astype("float32") * K[
                            physical + token, head // (query_heads // kv_heads), d
                        ].astype("float32")
                    score[0] *= 1 / math.sqrt(width)
                    new_maximum = T.max(maximum[0], score[0])
                    rescale = T.exp(maximum[0] - new_maximum)
                    probability = T.exp(score[0] - new_maximum)
                    denominator[0] = denominator[0] * rescale + probability
                    maximum[0] = new_maximum
                    for d in T.serial(width):
                        output[d] = (
                            output[d] * rescale
                            + probability
                            * V[physical + token, head // (query_heads // kv_heads), d]
                        )
            for d in T.serial(width):
                Output[segment, row, head, d] = T.if_then_else(
                    denominator[0] > 0, output[d] / denominator[0], 0
                )
            Statistics[segment, row, head, 0] = maximum[0]
            Statistics[segment, row, head, 1] = denominator[0]

    return main
