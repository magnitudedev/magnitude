"""Streaming attention with SIMD-owned matrix fragments and online softmax.

Queries, scores, probabilities, running statistics and output stay in registers.
Only bounded K/V tiles cross threadgroup memory. The physical-run/partial-output
contract is shared with the other attention realizations.
"""

import math

import tilelang.metal.language as T
from tilelang import tvm

from magnitude_engine.numerics.policy import floating
from magnitude_engine.platform.execution import DType


def _element(buffer, tile, element):
    return T.call_intrin(
        buffer.dtype, tvm.ir.Op.get("tl.simdgroup_element_get"), buffer.data, tile, element
    )


def _assign(buffer, tile, element, value):
    return T.evaluate(
        T.call_intrin(
            "handle", tvm.ir.Op.get("tl.simdgroup_element_set"), buffer.data, tile, element, value
        )
    )


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
    features = width // 8
    key_fragments = key_tile // 8
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
            lane = T.get_thread_binding() % 32
            warp = T.get_thread_binding() // 32
            # Each lane owns two adjacent columns in one matrix row. The four
            # lanes belonging to that row differ in bits 0 and 3 of their IDs.
            matrix_row = (lane // 16) * 4 + (lane % 8) // 2
            matrix_col = ((lane // 8) % 2) * 4 + (lane % 2) * 2
            row = block * query_tile + warp * 8 + matrix_row
            segment, partition = partial // partitions, partial % partitions
            first = partition * partition_chunks * key_tile
            physical = T.if_then_else(segments == 1, 0, Run[segment, 0])
            query = T.alloc_local((features * 64,), dtype.value, scope="metal.simdgroup")
            output = T.alloc_local((features * 64,), "float32", scope="metal.simdgroup")
            score = T.alloc_local((key_fragments * 64,), "float32", scope="metal.simdgroup")
            operand = T.alloc_local((64,), dtype.value, scope="metal.simdgroup")
            probability = T.alloc_local((key_fragments * 64,), dtype.value, scope="metal.simdgroup")
            kv = T.alloc_shared((key_tile, width), dtype.value)
            values = T.alloc_local((key_fragments * 2,), "float32")
            maximum = T.alloc_local((1,), "float32")
            denominator = T.alloc_local((1,), "float32")
            local_max = T.alloc_local((1,), "float32")
            local_sum = T.alloc_local((1,), "float32")
            maximum[0] = -3.402823466e38
            denominator[0] = 0
            for feature in T.unroll(features, explicit=True):
                T.make_filled_simdgroup_matrix(output.data, feature, T.float32(0), 8, 8)
                for element in T.unroll(2, explicit=True):
                    _assign(
                        query,
                        feature,
                        element,
                        T.if_then_else(
                            row < rows, Q[row, head, feature * 8 + matrix_col + element], 0
                        ),
                    )
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
                T.sync_threads()
                for j in T.unroll(key_fragments, explicit=True):
                    T.make_filled_simdgroup_matrix(score.data, j, T.float32(0), 8, 8)
                for feature in T.unroll(features, explicit=True):
                    for j in T.unroll(key_fragments, explicit=True):
                        T.simdgroup_load(
                            operand.data,
                            0,
                            T.access_ptr(kv[j * 8, feature * 8], "r"),
                            width,
                            8,
                            8,
                            True,
                        )
                        T.simdgroup_multiply_accumulate(
                            score.data, j, query.data, feature, operand.data, 0, score.data, j
                        )
                local_max[0] = maximum[0]
                for j in T.unroll(key_fragments, explicit=True):
                    for element in T.unroll(2, explicit=True):
                        token = first + chunk * key_tile + j * 8 + matrix_col + element
                        values[j * 2 + element] = T.if_then_else(
                            row < rows
                            and token < Run[segment, 2]
                            and token < segment_capacity
                            and Run[segment, 1] + token <= Positions[row],
                            _element(score, j, element) * (1 / math.sqrt(width)),
                            -3.402823466e38,
                        )
                        local_max[0] = T.max(local_max[0], values[j * 2 + element])
                local_max[0] = T.max(
                    local_max[0], T.call_extern("float32", "simd_shuffle_xor", local_max[0], 1)
                )
                local_max[0] = T.max(
                    local_max[0], T.call_extern("float32", "simd_shuffle_xor", local_max[0], 8)
                )
                alpha = T.exp(maximum[0] - local_max[0])
                local_sum[0] = 0
                for j in T.unroll(key_fragments, explicit=True):
                    for element in T.unroll(2, explicit=True):
                        values[j * 2 + element] = T.if_then_else(
                            values[j * 2 + element] > -3.402823466e38,
                            T.exp(values[j * 2 + element] - local_max[0]),
                            0,
                        )
                        local_sum[0] = local_sum[0] + values[j * 2 + element]
                        _assign(probability, j, element, values[j * 2 + element])
                local_sum[0] = local_sum[0] + T.call_extern(
                    "float32", "simd_shuffle_xor", local_sum[0], 1
                )
                local_sum[0] = local_sum[0] + T.call_extern(
                    "float32", "simd_shuffle_xor", local_sum[0], 8
                )
                denominator[0] = denominator[0] * alpha + local_sum[0]
                maximum[0] = local_max[0]
                for feature in T.unroll(features, explicit=True):
                    for element in T.unroll(2, explicit=True):
                        _assign(
                            output, feature, element, _element(output, feature, element) * alpha
                        )
                T.sync_threads()
                for i, d in T.Parallel(key_tile, width):
                    token = first + chunk * key_tile + i
                    kv[i, d] = T.if_then_else(
                        token < Run[segment, 2] and token < segment_capacity,
                        V[physical + token, head // (query_heads // kv_heads), d],
                        0,
                    )
                T.sync_threads()
                for feature in T.unroll(features, explicit=True):
                    for j in T.unroll(key_fragments, explicit=True):
                        T.simdgroup_load(
                            operand.data,
                            0,
                            T.access_ptr(kv[j * 8, feature * 8], "r"),
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
            if row < rows:
                if matrix_col == 0:
                    Statistics[partial, row, head, 0] = maximum[0]
                    Statistics[partial, row, head, 1] = denominator[0]
                for feature in T.unroll(features, explicit=True):
                    for element in T.unroll(2, explicit=True):
                        Output[partial, row, head, feature * 8 + matrix_col + element] = _element(
                            output, feature, element
                        ) / T.max(denominator[0], 1e-30)

    return main
