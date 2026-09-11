"""Causal attention over explicit segments in shared storage.

Each segment has a physical offset, logical start and visible length. Gaps
between segments are never read; partitioning and grouping do not relocate
history. Every realization produces normalized partial output plus the softmax
statistics that make those partials mergeable.
"""

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
    partitions: int = 1,
    head_tile: int = 1,
    query_tile: int = 8,
    key_tile: int = 32,
    feature_tile: int = 32,
    threads: int = 128,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    segment_capacity = capacity if segment_capacity is None else segment_capacity
    if (
        min(
            rows,
            query_heads,
            kv_heads,
            width,
            capacity,
            segment_capacity,
            segments,
            partitions,
            head_tile,
            query_tile,
            key_tile,
            feature_tile,
            threads,
        )
        <= 0
    ):
        raise ValueError("attention geometry must be positive")
    if (
        query_heads % kv_heads
        or (query_heads // kv_heads) % head_tile
        or any(n % 8 for n in (width, query_tile, key_tile, feature_tile))
    ):
        raise ValueError("invalid attention head or matrix tile geometry")
    if key_tile & (key_tile - 1):
        raise ValueError("attention reduction tile must be a power of two")

    if segment_capacity > capacity:
        raise ValueError("segment capacity exceeds its storage window")
    # Tile-aligned ranges partition each segment; visibility masks tails and gaps.
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
            T.ceildiv(rows * head_tile, query_tile),
            query_heads // head_tile,
            segments * partitions,
            threads=threads,
        ) as (
            block,
            head_group,
            partial,
        ):
            segment = partial // partitions
            partition = partial % partitions
            # A one-segment operand is already sliced to its physical origin.
            # Retain constant aligned addressing for the contiguous fast case.
            physical = T.if_then_else(segments == 1, 0, Run[segment, 0])
            q = T.alloc_shared((query_tile, feature_tile), dtype.value)
            k = T.alloc_shared((key_tile, feature_tile), dtype.value)
            v = T.alloc_shared((key_tile, feature_tile), "float32")
            scores = T.alloc_fragment((query_tile, key_tile), "float32")
            score_values = T.alloc_shared((query_tile, key_tile), "float32")
            probabilities = T.alloc_shared((query_tile, key_tile), "float32")
            reduction = T.alloc_shared((query_tile, key_tile), "float32")
            maximum = T.alloc_shared((query_tile,), "float32")
            denominator = T.alloc_shared((query_tile,), "float32")
            rescale = T.alloc_shared((query_tile,), "float32")
            output = T.alloc_shared((query_tile, width), "float32")
            product = T.alloc_fragment((query_tile, feature_tile), "float32")
            product_values = T.alloc_shared((query_tile, feature_tile), "float32")
            for i, d in T.Parallel(query_tile, width):
                output[i, d] = 0
            for i in T.Parallel(query_tile):
                maximum[i] = -3.402823466e38
                denominator[i] = 0
            T.sync_threads()
            for chunk in T.serial(
                T.max(
                    0,
                    T.min(
                        partition_chunks,
                        T.ceildiv(Run[segment, 2], key_tile) - partition * partition_chunks,
                    ),
                )
            ):
                T.clear(scores)
                for feature in T.serial(T.ceildiv(width, feature_tile)):
                    for i, d in T.Parallel(query_tile, feature_tile):
                        query = block * query_tile + i
                        row = query // head_tile
                        head = head_group * head_tile + query % head_tile
                        column = feature * feature_tile + d
                        if row < rows and column < width:
                            q[i, d] = Q[row, head, column]
                        else:
                            q[i, d] = 0
                    for j, d in T.Parallel(key_tile, feature_tile):
                        token = (partition * partition_chunks + chunk) * key_tile + j
                        column = feature * feature_tile + d
                        if token < Run[segment, 2] and token < segment_capacity and column < width:
                            k[j, d] = K[
                                physical + token,
                                head_group // (query_heads // (kv_heads * head_tile)),
                                column,
                            ]
                        else:
                            k[j, d] = 0
                    T.sync_threads()
                    T.gemm(q, k, scores, transpose_B=True)
                    T.sync_threads()
                T.copy(scores, score_values)
                T.sync_threads()
                for i, j in T.Parallel(query_tile, key_tile):
                    row = (block * query_tile + i) // head_tile
                    token = (partition * partition_chunks + chunk) * key_tile + j
                    if (
                        row < rows
                        and token < Run[segment, 2]
                        and token < segment_capacity
                        and Run[segment, 1] + token <= Positions[row]
                    ):
                        score_values[i, j] *= 1 / math.sqrt(width)
                    else:
                        score_values[i, j] = -3.402823466e38
                    reduction[i, j] = score_values[i, j]
                T.sync_threads()
                for step in T.unroll(int(math.log2(key_tile))):
                    for i, j in T.Parallel(query_tile, key_tile):
                        if j < (key_tile >> (step + 1)):
                            reduction[i, j] = T.max(
                                reduction[i, j], reduction[i, j + (key_tile >> (step + 1))]
                            )
                    T.sync_threads()
                for i in T.Parallel(query_tile):
                    new_maximum = T.max(maximum[i], reduction[i, 0])
                    rescale[i] = T.exp(maximum[i] - new_maximum)
                    maximum[i] = new_maximum
                T.sync_threads()
                for i, j in T.Parallel(query_tile, key_tile):
                    row = (block * query_tile + i) // head_tile
                    token = (partition * partition_chunks + chunk) * key_tile + j
                    if (
                        row < rows
                        and token < Run[segment, 2]
                        and token < segment_capacity
                        and Run[segment, 1] + token <= Positions[row]
                    ):
                        reduction[i, j] = T.exp(score_values[i, j] - maximum[i])
                    else:
                        reduction[i, j] = 0
                    probabilities[i, j] = reduction[i, j]
                T.sync_threads()
                for step in T.unroll(int(math.log2(key_tile))):
                    for i, j in T.Parallel(query_tile, key_tile):
                        if j < (key_tile >> (step + 1)):
                            reduction[i, j] += reduction[i, j + (key_tile >> (step + 1))]
                    T.sync_threads()
                for i in T.Parallel(query_tile):
                    denominator[i] = denominator[i] * rescale[i] + reduction[i, 0]
                T.sync_threads()
                for feature in T.serial(T.ceildiv(width, feature_tile)):
                    for j, d in T.Parallel(key_tile, feature_tile):
                        token = (partition * partition_chunks + chunk) * key_tile + j
                        column = feature * feature_tile + d
                        if token < Run[segment, 2] and token < segment_capacity and column < width:
                            v[j, d] = V[
                                physical + token,
                                head_group // (query_heads // (kv_heads * head_tile)),
                                column,
                            ]
                        else:
                            v[j, d] = 0
                    T.sync_threads()
                    T.clear(product)
                    T.gemm(probabilities, v, product)
                    T.copy(product, product_values)
                    T.sync_threads()
                    for i, d in T.Parallel(query_tile, feature_tile):
                        column = feature * feature_tile + d
                        if column < width:
                            output[i, column] = (
                                output[i, column] * rescale[i] + product_values[i, d]
                            )
                    T.sync_threads()
            for i, d in T.Parallel(query_tile, width):
                query = block * query_tile + i
                row = query // head_tile
                head = head_group * head_tile + query % head_tile
                if row < rows:
                    Output[partial, row, head, d] = T.if_then_else(
                        denominator[i] > 0, output[i, d] / denominator[i], 0
                    )
            for i in T.Parallel(query_tile):
                query = block * query_tile + i
                row = query // head_tile
                head = head_group * head_tile + query % head_tile
                if row < rows:
                    Statistics[partial, row, head, 0] = maximum[i]
                    Statistics[partial, row, head, 1] = denominator[i]

    return main
