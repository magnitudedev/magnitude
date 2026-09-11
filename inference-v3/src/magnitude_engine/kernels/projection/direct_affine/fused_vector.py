"""Fused gate/up projection over one canonical direct-affine allocation."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.precision import Precision, Rounding, native_sigmoid
from magnitude_engine.kernels.projection.direct_affine.layout import check
from magnitude_engine.kernels.projection.direct_affine.vector import _bf16
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    CodeInterpretation,
    WeightLayout,
    canonical_layout,
)


def gated_vector(
    M: int,
    N: int,
    K: int,
    layout: WeightLayout,
    *,
    capability: Capability,
    precision: Precision,
):
    representation, coefficients = check(layout)
    if capability.subgroup_width != 32:
        raise ValueError("the fused gate reduces across a 32-lane subgroup")
    if precision.rounding != Rounding.NATIVE_BF16:
        raise ValueError("the fused gate rounds the way NATIVE_BF16 does")
    if (
        representation.code.low_bits != 4
        or representation.code.high_bits
        or representation.code.interpretation != CodeInterpretation.UNSIGNED
        or coefficients.bias_dtype != DType.BF16
        or layout.logical_rows != 2 * N
        or layout.first_row
        or layout.columns != K
        or K % representation.group
        or layout.nbytes % 4
    ):
        raise ValueError("invalid direct-affine gated geometry")

    resident = canonical_layout(representation, layout.elements)
    assert resident.biases is not None
    words = layout.nbytes // 4
    group = representation.group

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(N, M, threads=32) as (out, token):
            lane = T.get_thread_binding()
            partial = T.alloc_local((2,), "float32")
            dot = T.alloc_local((2,), "float32")
            bias_sum = T.alloc_local((1,), "float32")
            partial[0] = 0
            partial[1] = 0
            for i in T.serial(T.ceildiv(K, 512)):
                base = (i * 32 + lane) * 16
                dot[0] = 0
                dot[1] = 0
                bias_sum[0] = 0
                for j in T.unroll(4, explicit=True):
                    column = base + j * 4
                    if column < K:
                        x0 = A[token, column].astype("float32")
                        x1 = A[token, column + 1].astype("float32")
                        x2 = A[token, column + 2].astype("float32")
                        x3 = A[token, column + 3].astype("float32")
                        xs = (x0 + x1).astype("bfloat16").astype("float32")
                        xs2 = (xs + x2).astype("bfloat16").astype("float32")
                        bias_sum[0] += (xs2 + x3).astype("bfloat16").astype("float32")
                        for branch in T.unroll(2, explicit=True):
                            backing_row = branch * N + out
                            element = backing_row * K + column
                            packed = B[(resident.low + element // 2) // 4]
                            word = (packed >> ((element % 8) * 4)) & T.uint32(65535)
                            dot[branch] += (
                                x0 * (word & T.uint32(15)).astype("float32")
                                + x1 * ((word >> 4) & T.uint32(15)).astype("float32")
                                + x2 * ((word >> 8) & T.uint32(15)).astype("float32")
                                + x3 * ((word >> 12) & T.uint32(15)).astype("float32")
                            )
                if base < K:
                    for branch in T.unroll(2, explicit=True):
                        backing_row = branch * N + out
                        coefficient = (backing_row * (K // group) + base // group) * 2
                        scale = _bf16(B, resident.scales + coefficient)
                        bias = _bf16(B, resident.biases + coefficient)
                        partial[branch] += dot[branch] * scale + bias_sum[0] * bias
            gate = T.warp_reduce_sum(partial[0]).astype("bfloat16").astype("float32")
            up = T.warp_reduce_sum(partial[1]).astype("bfloat16").astype("float32")
            if lane == 0:
                activated = (
                    (gate * native_sigmoid(gate).astype("float32"))
                    .astype("bfloat16")
                    .astype("float32")
                )
                C[token, out] = activated * up

    return main
