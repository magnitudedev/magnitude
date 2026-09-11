"""Fused gate/up contraction over the canonical Q4/Q5 hierarchy."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.precision import Precision, Rounding, native_sigmoid
from magnitude_engine.kernels.projection.packed_k import _byte, _half, _packed
from magnitude_engine.weights.representation import (
    HierarchicalCoefficients,
    WeightLayout,
    canonical_layout,
    is_scale_min_hierarchy,
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
    representation = layout.representation
    if capability.subgroup_width != 32:
        raise ValueError("the hierarchical gate reduces across a 32-lane subgroup")
    if precision.rounding != Rounding.NATIVE_BF16:
        raise ValueError("the hierarchical gate rounds the way NATIVE_BF16 does")
    if (
        not is_scale_min_hierarchy(representation)
        or layout.logical_rows != 2 * N
        or layout.first_row
        or layout.columns != K
        or K % 256
        or layout.nbytes % 4
    ):
        raise ValueError("the hierarchical gate requires canonical Q4_K or Q5_K")
    coefficients = representation.coefficients
    assert isinstance(coefficients, HierarchicalCoefficients)

    resident = canonical_layout(representation, layout.elements)
    assert resident.biases is not None
    assert resident.super_scale is not None
    assert resident.super_bias is not None
    words = layout.nbytes // 4

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(N, M, threads=32) as (out, row):
            lane = T.get_thread_binding()
            group = lane // 4
            partial = T.alloc_local((2,), "float32")
            dot = T.alloc_local((2,), "float32")
            sums = T.alloc_local((1,), "float32")
            values = T.alloc_local((8,), "float32")
            T.clear(partial)
            for chunk in T.serial(K // 256):
                sums[0] = 0
                for j in T.unroll(8, explicit=True):
                    values[j] = A[row, chunk * 256 + lane * 8 + j].astype("float32")
                    sums[0] += values[j]
                for branch in T.unroll(2, explicit=True):
                    element = (branch * N + out) * K + chunk * 256
                    base = (element // 256) * resident.tile_bytes
                    low = B[(base + resident.low) // 4 + lane]
                    high = T.uint32(0)
                    if representation.code.high_bits:
                        high = _byte(B, base + resident.high + lane)
                    scale = _packed(B, base + resident.scales, 6, group)
                    bias = _packed(B, base + resident.biases, 6, group)
                    super_scale = _half(B, base + resident.super_scale)
                    super_bias = _half(B, base + resident.super_bias)
                    dot[branch] = 0
                    for j in T.unroll(8, explicit=True):
                        code = (low >> (j * 4)) & T.uint32(15)
                        if representation.code.high_bits:
                            code = code | (((high >> j) & T.uint32(1)) << 4)
                        dot[branch] += values[j] * code.astype("float32")
                    partial[branch] += (
                        super_scale * scale * dot[branch] - super_bias * bias * sums[0]
                    )
            gate = T.warp_reduce_sum(partial[0]).astype("bfloat16").astype("float32")
            up = T.warp_reduce_sum(partial[1]).astype("bfloat16").astype("float32")
            if lane == 0:
                activated = (
                    (gate * native_sigmoid(gate).astype("float32"))
                    .astype("bfloat16")
                    .astype("float32")
                )
                C[row, out] = activated * up

    return main
