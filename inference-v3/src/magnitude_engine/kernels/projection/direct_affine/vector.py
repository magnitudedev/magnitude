"""GEMV over a canonical direct-affine allocation."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.projection.direct_affine.layout import check
from magnitude_engine.kernels.projection.layout import output_index
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    CodeInterpretation,
    WeightLayout,
    canonical_layout,
)


@T.macro
def _bf16(words, byte):
    bits = ((words[byte // 4] >> ((byte % 4) * 8)) & T.uint32(65535)).astype("uint16")
    return T.reinterpret(bits, "bfloat16").astype("float32")


def vector(
    rows: int,
    widths: tuple[int, ...],
    inputs: int,
    layout: WeightLayout,
    *,
    capability: Capability,
    dtype: DType = DType.BF16,
    output_dtype: DType = DType.BF16,
):
    representation, coefficients = check(layout)
    outputs = sum(widths)
    if capability.subgroup_width != 32:
        raise ValueError("the affine GEMV reduces across a 32-lane subgroup")
    if (
        min(rows, inputs) <= 0
        or not widths
        or any(width <= 0 for width in widths)
        or outputs != layout.logical_rows
        or inputs != layout.columns
        or representation.code.low_bits != 4
        or representation.code.high_bits
        or representation.code.interpretation != CodeInterpretation.UNSIGNED
        or coefficients.bias_dtype != DType.BF16
        or inputs % representation.group
        or layout.nbytes % 4
    ):
        raise ValueError("invalid direct-affine GEMV geometry")

    resident = canonical_layout(representation, layout.elements)
    assert resident.biases is not None
    words = layout.nbytes // 4
    group = representation.group
    output_at = output_index(rows, widths)
    pack = 16 if outputs % 8 == 0 else 8

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((rows * outputs,), output_dtype.value),
    ):
        with T.Kernel(outputs, rows, threads=32) as (out, token):
            lane = T.get_thread_binding()
            backing_row = layout.first_row + out
            partial = T.alloc_local((1,), "float32")
            dot = T.alloc_local((1,), "float32")
            bias_sum = T.alloc_local((1,), "float32")
            partial[0] = 0
            for i in T.serial(T.ceildiv(inputs, 32 * pack)):
                base = (i * 32 + lane) * pack
                dot[0] = 0
                bias_sum[0] = 0
                for j in T.serial(pack // 4):
                    k = base + j * 4
                    if k < inputs:
                        element = backing_row * inputs + k
                        packed = B[(resident.low + element // 2) // 4]
                        word = (packed >> ((element % 8) * 4)) & T.uint32(65535)
                        x0 = A[token, k].astype("float32")
                        x1 = A[token, k + 1].astype("float32")
                        x2 = A[token, k + 2].astype("float32")
                        x3 = A[token, k + 3].astype("float32")
                        xs = (x0 + x1).astype("bfloat16").astype("float32")
                        xs2 = (xs + x2).astype("bfloat16").astype("float32")
                        bias_sum[0] += (xs2 + x3).astype("bfloat16").astype("float32")
                        dot[0] += (
                            x0 * (word & T.uint32(15)).astype("float32")
                            + x1 * ((word >> 4) & T.uint32(15)).astype("float32")
                            + x2 * ((word >> 8) & T.uint32(15)).astype("float32")
                            + x3 * ((word >> 12) & T.uint32(15)).astype("float32")
                        )
                if base < inputs:
                    coefficient = (backing_row * (inputs // group) + base // group) * 2
                    scale = _bf16(B, resident.scales + coefficient)
                    bias = _bf16(B, resident.biases + coefficient)
                    partial[0] += dot[0] * scale + bias_sum[0] * bias
            total = T.warp_reduce_sum(partial[0])
            if lane == 0:
                C[output_at(token, out)] = total.astype("bfloat16")

    return main
