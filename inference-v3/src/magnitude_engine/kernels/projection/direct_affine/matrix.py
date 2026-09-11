"""Tiled contraction over a canonical direct-affine allocation."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.projection.direct_affine.layout import check
from magnitude_engine.kernels.projection.direct_affine.vector import _bf16
from magnitude_engine.kernels.projection.layout import output_index
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    CodeInterpretation,
    WeightLayout,
    canonical_layout,
)


def matrix(
    M: int,
    widths: tuple[int, ...],
    K: int,
    layout: WeightLayout,
    PARTS: int,
    BM: int,
    BN: int,
    BK: int,
    PAD: int,
    *,
    capability: Capability,
    output_dtype: DType = DType.BF16,
):
    representation, coefficients = check(layout)
    if not capability.matrix_instructions:
        raise ValueError("the tiled affine contraction requires matrix hardware")
    N = sum(widths)
    if (
        representation.code.low_bits != 4
        or representation.code.high_bits
        or representation.code.interpretation != CodeInterpretation.UNSIGNED
        or coefficients.bias_dtype != DType.BF16
        or N != layout.logical_rows
        or K != layout.columns
        or K % (PARTS * representation.group)
        or (K // PARTS) % BK
        or layout.nbytes % 4
    ):
        raise ValueError("invalid direct-affine matrix geometry")

    resident = canonical_layout(representation, layout.elements)
    assert resident.biases is not None
    words = layout.nbytes // 4
    group = representation.group
    output_at = output_index(M, widths)
    span = K // PARTS

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((PARTS, M * N), output_dtype.value),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), PARTS, threads=128) as (
            bx,
            by,
            part,
        ):
            tid = T.get_thread_binding()
            inputs = T.alloc_shared((BM, BK + PAD), "bfloat16")
            weights = T.alloc_shared((BN, BK + PAD), "bfloat16")
            result = T.alloc_shared((BM, BN), "float32")
            accum = T.alloc_fragment((BM, BN), "float32")
            T.clear(accum)
            for block in T.serial(span // BK):
                for i, j in T.Parallel(BM, BK):
                    if by * BM + i < M:
                        inputs[i, j] = A[by * BM + i, part * span + block * BK + j]
                    else:
                        inputs[i, j] = T.cast(0, "bfloat16")
                for word_index in T.serial(T.ceildiv(BN * BK // 8, 128)):
                    index = word_index * 128 + tid
                    row = index // (BK // 8)
                    col = index % (BK // 8) * 8
                    if row < BN:
                        if bx * BN + row < N:
                            backing_row = layout.first_row + bx * BN + row
                            global_col = part * span + block * BK + col
                            element = backing_row * K + global_col
                            packed = B[(resident.low + element // 2) // 4]
                            coefficient = (backing_row * (K // group) + global_col // group) * 2
                            scale = _bf16(B, resident.scales + coefficient)
                            bias = _bf16(B, resident.biases + coefficient)
                            for offset in T.unroll(8, explicit=True):
                                code = (packed >> T.uint32(offset * 4)) & T.uint32(15)
                                weights[row, col + offset] = code.astype("float32") * scale + bias
                        else:
                            for offset in T.unroll(8, explicit=True):
                                weights[row, col + offset] = T.cast(0, "bfloat16")
                T.sync_threads()
                T.gemm(inputs[:, :BK], weights[:, :BK], accum, transpose_B=True)
                T.sync_threads()
            T.sync_threads()
            T.copy(accum, result)
            T.sync_threads()
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M and bx * BN + j < N:
                    C[
                        part,
                        output_at(by * BM + i, bx * BN + j)
                        if PARTS == 1
                        else (by * BM + i) * N + bx * BN + j,
                    ] = result[i, j].astype("bfloat16")

    return main
