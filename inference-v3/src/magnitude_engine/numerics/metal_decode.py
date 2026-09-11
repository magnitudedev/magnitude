"""Bounded matrix partitions for few-query grouped attention.

One SIMD group owns up to eight (query row, query head) pairs sharing a KV head.
It retains complete queries and outputs in matrix registers. Scores for a bounded
history partition stay in shared memory and are normalized once. A single shared
KV tile also stages query input and final output, avoiding feature-wise barriers.
"""

import math

import tilelang.metal.language as T

from magnitude_engine.numerics.policy import floating
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
    partitions: int,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    if (
        min(rows, query_heads, kv_heads, width, capacity, segments, segment_capacity, partitions)
        <= 0
    ):
        raise ValueError("attention geometry must be positive")
    if query_heads % kv_heads or width % 8 or width > 256 or segment_capacity > capacity:
        raise ValueError("invalid SIMD matrix attention geometry")
    floating(dtype)
    floating(output_dtype)
    head_tile = query_heads // kv_heads
    query_tile, key_tile = 8, 16
    features = width // 8
    chunks = (segment_capacity + partitions * key_tile - 1) // (partitions * key_tile)
    span = chunks * key_tile
    if span > 256:
        raise ValueError("attention partition exceeds bounded score storage")

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
            T.ceildiv(rows * head_tile, query_tile), kv_heads, segments * partitions, threads=32
        ) as (block, kv_head, partial):
            lane = T.get_thread_binding()
            segment = partial // partitions
            first = partial % partitions * span
            physical = T.if_then_else(segments == 1, 0, Run[segment, 0])
            count = T.max(0, T.min(span, T.min(Run[segment, 2], segment_capacity) - first))
            query = T.alloc_local((features * 64,), "float32", scope="metal.simdgroup")
            output = T.alloc_local((features * 64,), "float32", scope="metal.simdgroup")
            score = T.alloc_local((2 * 64,), "float32", scope="metal.simdgroup")
            operand = T.alloc_local((64,), "float32", scope="metal.simdgroup")
            probability = T.alloc_local((2 * 64,), "float32", scope="metal.simdgroup")
            # Compact global storage is widened only in this bounded stage.
            # Decode retains FP32 probability/value arithmetic and accumulation.
            stage = T.alloc_shared((key_tile, width), "float32")
            scores = T.alloc_shared((query_tile, span), "float32")
            maximum = T.alloc_local((1,), "float32")
            total = T.alloc_local((1,), "float32")
            for i, d in T.Parallel(query_tile, width):
                q = block * query_tile + i
                stage[i, d] = T.if_then_else(
                    q < rows * head_tile,
                    Q[q // head_tile, kv_head * head_tile + q % head_tile, d],
                    0,
                )
            T.sync_threads()
            for feature in T.unroll(features, explicit=True):
                T.make_filled_simdgroup_matrix(output.data, feature, T.float32(0), 8, 8)
                T.simdgroup_load(
                    query.data,
                    feature,
                    T.access_ptr(stage[0, feature * 8], "r"),
                    width,
                    8,
                    8,
                    False,
                )
            T.sync_threads()
            for chunk in T.serial(chunks):
                for i, d in T.Parallel(key_tile, width):
                    token = chunk * key_tile + i
                    stage[i, d] = T.if_then_else(
                        token < count, K[physical + first + token, kv_head, d], 0
                    )
                T.sync_threads()
                for j in T.unroll(2, explicit=True):
                    T.make_filled_simdgroup_matrix(score.data, j, T.float32(0), 8, 8)
                for feature in T.unroll(features, explicit=True):
                    for j in T.unroll(2, explicit=True):
                        T.simdgroup_load(
                            operand.data,
                            0,
                            T.access_ptr(stage[j * 8, feature * 8], "r"),
                            width,
                            8,
                            8,
                            True,
                        )
                        T.simdgroup_multiply_accumulate(
                            score.data, j, query.data, feature, operand.data, 0, score.data, j
                        )
                for j in T.unroll(2, explicit=True):
                    T.simdgroup_store(
                        score.data,
                        j,
                        T.access_ptr(scores[0, chunk * key_tile + j * 8], "w"),
                        span,
                        8,
                        8,
                        False,
                    )
                T.sync_threads()
            # Each lane owns a strided subset of a query's partition. Normalize
            # once, including causal masks and empty partitions, before P @ V.
            for i in T.unroll(query_tile, explicit=True):
                q = block * query_tile + i
                maximum[0] = -3.402823466e38
                for step in T.unroll(T.ceildiv(span, 32), explicit=True):
                    token = step * 32 + lane
                    if token < span:
                        value = T.alloc_local((1,), "float32")
                        value[0] = -3.402823466e38
                        if q < rows * head_tile and token < count:
                            if Run[segment, 1] + first + token <= Positions[q // head_tile]:
                                value[0] = scores[i, token] * (1 / math.sqrt(width))
                        scores[i, token] = value[0]
                        maximum[0] = T.max(maximum[0], value[0])
                m = T.warp_reduce_max(maximum[0])
                total[0] = 0
                for step in T.unroll(T.ceildiv(span, 32), explicit=True):
                    token = step * 32 + lane
                    if token < span:
                        p = T.if_then_else(
                            scores[i, token] > -3.402823466e38, T.exp(scores[i, token] - m), 0
                        )
                        scores[i, token] = p
                        total[0] += p
                mass = T.warp_reduce_sum(total[0])
                for step in T.unroll(T.ceildiv(span, 32), explicit=True):
                    token = step * 32 + lane
                    if token < span:
                        scores[i, token] = scores[i, token] / T.max(mass, 1e-30)
                if lane == 0 and q < rows * head_tile:
                    Statistics[partial, q // head_tile, kv_head * head_tile + q % head_tile, 0] = m
                    Statistics[partial, q // head_tile, kv_head * head_tile + q % head_tile, 1] = (
                        mass
                    )
            T.sync_threads()
            for chunk in T.serial(chunks):
                for i, d in T.Parallel(key_tile, width):
                    token = chunk * key_tile + i
                    stage[i, d] = T.if_then_else(
                        token < count, V[physical + first + token, kv_head, d], 0
                    )
                T.sync_threads()
                for j in T.unroll(2, explicit=True):
                    T.simdgroup_load(
                        probability.data,
                        j,
                        T.access_ptr(scores[0, chunk * key_tile + j * 8], "r"),
                        span,
                        8,
                        8,
                        False,
                    )
                for feature in T.unroll(features, explicit=True):
                    for j in T.unroll(2, explicit=True):
                        T.simdgroup_load(
                            operand.data,
                            0,
                            T.access_ptr(stage[j * 8, feature * 8], "r"),
                            width,
                            8,
                            8,
                            False,
                        )
                        T.simdgroup_multiply_accumulate(
                            output.data,
                            feature,
                            probability.data,
                            j,
                            operand.data,
                            0,
                            output.data,
                            feature,
                        )
                T.sync_threads()
            for feature in T.unroll(features, explicit=True):
                T.simdgroup_store(
                    output.data,
                    feature,
                    T.access_ptr(stage[0, feature * 8], "w"),
                    width,
                    8,
                    8,
                    False,
                )
            T.sync_threads()
            for i, d in T.Parallel(query_tile, width):
                q = block * query_tile + i
                if q < rows * head_tile:
                    Output[partial, q // head_tile, kv_head * head_tile + q % head_tile, d] = stage[
                        i, d
                    ]

    return main
