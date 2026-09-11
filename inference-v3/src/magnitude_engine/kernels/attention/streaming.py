"""Streaming attention with logical TileLang fragments and online softmax.

Queries, scores, probabilities, running statistics and output stay in registers.
Only bounded K/V tiles cross threadgroup memory. The physical-run/partial-output
contract is shared with the other attention realizations.
"""

import math

import tilelang.language as T

from magnitude_engine.kernels.precision import floating
from magnitude_engine.platform.execution import DType


def run_attention(
    rows: int,
    query_heads: int,
    kv_heads: int,
    width: int,
    capacity: int,
    *,
    segments: int,
    segment_capacity: int,
    partitions: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    if (
        min(rows, query_heads, kv_heads, width, capacity, segments, segment_capacity, partitions)
        <= 0
    ):
        raise ValueError("attention geometry must be positive")
    if query_heads % kv_heads or width % 8 or segment_capacity > capacity:
        raise ValueError("invalid streaming attention geometry")
    floating(dtype)
    floating(output_dtype)
    query_tile, key_tile, groups = 32, 32, 4
    partition_chunks = (segment_capacity + partitions * key_tile - 1) // (partitions * key_tile)

    @T.prim_func
    def main(
        Q: T.Tensor((rows, query_heads, width), dtype.value),
        K: T.Tensor((capacity, kv_heads, width), dtype.value),
        V: T.Tensor((capacity, kv_heads, width), dtype.value),
        Positions: T.Tensor((rows,), "int32"),
        Run: T.Tensor((segments, 3), "int32"),
        Output: T.Tensor((segments * partitions, rows, query_heads, width), output_dtype.value),
        Statistics: T.Tensor((segments * partitions, rows, query_heads, 2), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(rows, query_tile), query_heads, segments * partitions, threads=groups * 32
        ) as (block, head, partial):
            segment, partition = partial // partitions, partial % partitions
            first = partition * partition_chunks * key_tile
            physical = T.if_then_else(segments == 1, 0, Run[segment, 0])
            query = T.alloc_fragment((query_tile, width), dtype.value)
            output = T.alloc_fragment((query_tile, width), "float32")
            score = T.alloc_fragment((query_tile, key_tile), "float32")
            probability = T.alloc_fragment((query_tile, key_tile), dtype.value)
            kv = T.alloc_shared((key_tile, width), dtype.value)
            maximum = T.alloc_fragment((query_tile,), "float32")
            previous = T.alloc_fragment((query_tile,), "float32")
            denominator = T.alloc_fragment((query_tile,), "float32")
            local_sum = T.alloc_fragment((query_tile,), "float32")
            alpha = T.alloc_fragment((query_tile,), "float32")
            T.fill(maximum, -3.402823466e38)
            T.clear(denominator)
            T.clear(output)
            for i, d in T.Parallel(query_tile, width):
                row = block * query_tile + i
                query[i, d] = T.if_then_else(row < rows, Q[row, head, d], 0)
            for chunk in T.serial(
                T.max(0, T.min(partition_chunks, T.ceildiv(Run[segment, 2] - first, key_tile)))
            ):
                for i, d in T.Parallel(key_tile, width):
                    token = first + chunk * key_tile + i
                    kv[i, d] = T.if_then_else(
                        token < Run[segment, 2] and token < segment_capacity,
                        K[physical + token, head // (query_heads // kv_heads), d],
                        0,
                    )
                T.gemm(
                    query,
                    kv,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(query_tile, key_tile):
                    row = block * query_tile + i
                    token = first + chunk * key_tile + j
                    score[i, j] = T.if_then_else(
                        row < rows
                        and token < Run[segment, 2]
                        and token < segment_capacity
                        and Run[segment, 1] + token <= Positions[row],
                        score[i, j] * (1 / math.sqrt(width)),
                        -3.402823466e38,
                    )
                T.copy(maximum, previous)
                T.reduce_max(score, maximum, dim=1, clear=False)
                for i in T.Parallel(query_tile):
                    alpha[i] = T.exp(previous[i] - maximum[i])
                for i, j in T.Parallel(query_tile, key_tile):
                    score[i, j] = T.if_then_else(
                        score[i, j] > -3.402823466e38,
                        T.exp(score[i, j] - maximum[i]),
                        0,
                    )
                    probability[i, j] = T.cast(score[i, j], dtype.value)
                T.reduce_sum(score, local_sum, dim=1)
                for i in T.Parallel(query_tile):
                    denominator[i] = denominator[i] * alpha[i] + local_sum[i]
                for i, d in T.Parallel(query_tile, width):
                    output[i, d] *= alpha[i]
                for i, d in T.Parallel(key_tile, width):
                    token = first + chunk * key_tile + i
                    kv[i, d] = T.if_then_else(
                        token < Run[segment, 2] and token < segment_capacity,
                        V[physical + token, head // (query_heads // kv_heads), d],
                        0,
                    )
                T.gemm(probability, kv, output, policy=T.GemmWarpPolicy.FullRow)
            for i in T.Parallel(query_tile):
                row = block * query_tile + i
                if row < rows:
                    Statistics[partial, row, head, 0] = maximum[i]
                    Statistics[partial, row, head, 1] = denominator[i]
            for i, d in T.Parallel(query_tile, width):
                row = block * query_tile + i
                if row < rows:
                    Output[partial, row, head, d] = output[i, d] / T.max(denominator[i], 1e-30)

    return main
