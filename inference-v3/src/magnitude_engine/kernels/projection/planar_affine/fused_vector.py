"""Gate and up projections plus their activation, in one pass over the input."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.precision import Precision, Rounding, native_sigmoid
from magnitude_engine.kernels.projection.planar_affine.layout import check
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import PlanarAffine


def gated_vector(
    M,
    N,
    K,
    representation: PlanarAffine,
    *,
    capability: Capability,
    precision: Precision,
):
    """One pass over the input computes both branches and the gate product.

    A fusion is a candidate of the composite operation, never a special case
    inside one of its components.
    """
    check(representation)
    if capability.subgroup_width != 32:
        raise ValueError("the fused gate reduces across a 32-lane subgroup")
    if precision.rounding != Rounding.NATIVE_BF16:
        raise ValueError("the fused gate rounds the way NATIVE_BF16 does")
    if representation.coefficient_dtype != DType.BF16 or representation.high_bits:
        raise ValueError("this fusion reads row-addressed BF16 coefficient planes")
    group = representation.group

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((2 * N, K // 8), "uint32"),
        C: T.Tensor((2 * N, K // group), "bfloat16"),
        D: T.Tensor((2 * N, K // group), "bfloat16"),
        E: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(N, M, threads=32) as (row, token):
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
                            word = (
                                B[branch * N + row, column // 8] >> ((column % 8) * 4)
                            ) & T.uint32(65535)
                            dot[branch] += (
                                x0 * (word & T.uint32(15)).astype("float32")
                                + x1 * ((word >> 4) & T.uint32(15)).astype("float32")
                                + x2 * ((word >> 8) & T.uint32(15)).astype("float32")
                                + x3 * ((word >> 12) & T.uint32(15)).astype("float32")
                            )
                if base < K:
                    for branch in T.unroll(2, explicit=True):
                        partial[branch] += dot[branch] * C[
                            branch * N + row, base // group
                        ].astype("float32") + bias_sum[0] * D[
                            branch * N + row, base // group
                        ].astype("float32")
            gate = T.warp_reduce_sum(partial[0]).astype("bfloat16").astype("float32")
            up = T.warp_reduce_sum(partial[1]).astype("bfloat16").astype("float32")
            if lane == 0:
                activated = (
                    (gate * native_sigmoid(gate).astype("float32"))
                    .astype("bfloat16")
                    .astype("float32")
                )
                E[token, row] = activated * up

    return main
