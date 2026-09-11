"""Compact encoded planes and a register fold shared by affine K formats.

One allocation contains low nibbles, exact FP32 group parameters, then high bits.
A SIMD group owns four outputs and reuses prepared input packs across all four.
"""

import tilelang.language as T

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.numerics.encoded_groups import interpretation
from magnitude_engine.numerics.encoded_layout import EncodedLayout, affine_geometry, storage_bytes
from magnitude_engine.platform.execution import DType


def pack(blocks: int, elements: int, encoding: Encoding):
    """Place a bounded source chunk into final planes at a runtime element offset."""
    if blocks <= 0 or elements < blocks * 256 or elements % 256:
        raise ValueError("packing requires complete K superblocks")
    geometry = affine_geometry(encoding)
    group_elements, high_bits, zero_point = (
        geometry.group_elements,
        geometry.high_bits,
        geometry.zero_point,
    )
    groups = blocks * 256 // group_elements
    words = storage_bytes(elements, encoding, EncodedLayout.AFFINE_PLANES) // 4
    parameters, code = interpretation(encoding)

    @T.prim_func
    def main(
        A: T.Tensor((blocks * (encoding.block_bytes // 2),), "uint16"),
        B: T.Tensor((words,), "uint32"),
        Offset: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(T.ceildiv(groups, 128), threads=128) as block:
            group = block * 128 + T.get_thread_binding()
            low = T.alloc_local((1,), "uint32")
            high = T.alloc_local((1,), "uint32")
            if group < groups:
                first = Offset[0] + group * group_elements
                scale, bias = parameters(A, group * group_elements)
                B[elements // 8 + first // group_elements] = T.reinterpret(scale, "uint32")
                if zero_point == 0:
                    B[elements // 8 + elements // group_elements + first // group_elements] = (
                        T.reinterpret(bias, "uint32")
                    )
                high[0] = 0
                for packet in T.unroll(group_elements // 8, explicit=True):
                    low[0] = 0
                    for j in T.unroll(8, explicit=True):
                        value = (
                            code(A, group * group_elements + packet * 8 + j) + zero_point
                        ).astype("uint32")
                        low[0] |= (value & T.uint32(15)) << T.uint32(j * 4)
                        if high_bits:
                            high[0] |= (value >> 4) << T.uint32((packet * 8 + j) * high_bits)
                    B[first // 8 + packet] = low[0]
                if high_bits:
                    B[elements * 3 // 16 + first // group_elements] = high[0]

    return main


def vector(
    rows: int,
    outputs: int,
    inputs: int,
    encoding: Encoding,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    if min(rows, outputs, inputs) <= 0 or inputs % 512:
        raise ValueError("complete 512-coordinate folds required")
    geometry = affine_geometry(encoding)
    group_elements, high_bits, zero_point = (
        geometry.group_elements,
        geometry.high_bits,
        geometry.zero_point,
    )
    elements = outputs * inputs
    groups = elements // group_elements
    words = storage_bytes(elements, encoding, EncodedLayout.AFFINE_PLANES) // 4

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(T.ceildiv(outputs, 8), rows, threads=64) as (block, row):
            thread = T.get_thread_binding()
            lane = thread % 32
            first = block * 8 + thread // 32 * 4
            values = T.alloc_local((16,), "float32")
            high_values = T.alloc_local((16,), "float32")
            accum = T.alloc_local((4,), "float32")
            total = T.alloc_local((1,), "float32")
            dot = T.alloc_local((1,), "float32")
            T.clear(accum)
            for step in T.serial(inputs // 512):
                k = step * 512 + lane * 16
                total[0] = 0
                for j in T.unroll(16, explicit=True):
                    x = A[row, k + j].astype("float32")
                    total[0] += x
                    values[j] = x * (1.0 / (1 << (4 * (j % 4))))
                    if high_bits:
                        high_values[j] = x * (16.0 / (1 << (j * high_bits)))
                for out in T.unroll(4, explicit=True):
                    if first + out < outputs:
                        g = (first + out) * (inputs // group_elements) + k // group_elements
                        scale = T.reinterpret(B[elements // 8 + g], "float32")
                        dot[0] = 0
                        for packet in T.unroll(2, explicit=True):
                            packed = B[(first + out) * (inputs // 8) + k // 8 + packet]
                            for half in T.unroll(2, explicit=True):
                                word = (packed >> T.uint32(half * 16)) & T.uint32(65535)
                                for j in T.unroll(4, explicit=True):
                                    dot[0] += values[packet * 8 + half * 4 + j] * (
                                        word & T.uint32(15 << (j * 4))
                                    ).astype("float32")
                        if high_bits:
                            high_word = B[elements * 3 // 16 + g] >> T.uint32(
                                (k % group_elements) * high_bits
                            )
                            for j in T.unroll(16, explicit=True):
                                coefficient = (
                                    high_word
                                    & (T.uint32((1 << high_bits) - 1) << T.uint32(j * high_bits))
                                ).astype("float32") - (
                                    T.uint32(zero_point // 16) << T.uint32(j * high_bits)
                                ).astype("float32")
                                dot[0] += high_values[j] * coefficient
                        if zero_point:
                            accum[out] += scale * dot[0]
                        else:
                            bias = T.reinterpret(B[elements // 8 + groups + g], "float32")
                            accum[out] += scale * dot[0] + bias * total[0]
            for out in T.unroll(4, explicit=True):
                result = T.warp_reduce_sum(accum[out])
                if lane == 0 and first + out < outputs:
                    C[row, first + out] = result

    return main
