"""Group-wise encoded contractions: payload dot products precede shared scales.

A SIMD group consumes 256 adjacent logical weights per iteration. Each lane owns
eight coordinates within one quantization group, so it loads that group's scale
once. Resident storage stays in the original GGUF encoding.
"""

import tilelang.language as T

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.numerics.encoded_groups import interpretation
from magnitude_engine.platform.execution import DType


def projection(
    rows,
    outputs,
    inputs,
    encoding: Encoding,
    *,
    row_tile=1,
    output_tile=4,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    if encoding not in (Encoding.Q6_K, Encoding.IQ4_XS, Encoding.Q8_0, Encoding.F16):
        raise ValueError("group contraction requires a supported symmetric or dense encoding")
    if min(rows, outputs, inputs, row_tile, output_tile) <= 0 or inputs % encoding.block_elements:
        raise ValueError("invalid group contraction geometry")
    parameters, payload = interpretation(encoding)
    size = outputs * inputs // encoding.block_elements * encoding.block_bytes // 2

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint16"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile), T.ceildiv(rows, row_tile), threads=32 * output_tile
        ) as (block, row_group):
            tid = T.get_thread_binding()
            lane = tid % 32
            out = block * output_tile + tid // 32
            accum = T.alloc_local((row_tile,), "float32")
            dot = T.alloc_local((row_tile,), "float32")
            T.clear(accum)
            for chunk in T.serial(T.ceildiv(inputs, 256)):
                first = chunk * 256 + lane * 8
                T.clear(dot)
                if out < outputs and first < inputs:
                    scale, _ = parameters(B, out * inputs + first)
                    for j in T.unroll(8, explicit=True):
                        k = first + j
                        if k < inputs:
                            code = payload(B, out * inputs + k)
                            for r in T.unroll(row_tile, explicit=True):
                                row = row_group * row_tile + r
                                if row < rows:
                                    dot[r] += A[row, k].astype("float32") * code
                    for r in T.unroll(row_tile, explicit=True):
                        accum[r] += dot[r] * scale
            for r in T.unroll(row_tile, explicit=True):
                total = T.warp_reduce_sum(accum[r])
                row = row_group * row_tile + r
                if lane == 0 and out < outputs and row < rows:
                    C[row, out] = total

    return main
