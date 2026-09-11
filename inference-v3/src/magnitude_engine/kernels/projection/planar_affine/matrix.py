"""Tiled affine contraction for prefill over row-addressed coefficient planes."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.projection.planar_affine.layout import check, output_index
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import PlanarAffine


def matrix(
    M,
    widths: tuple[int, ...],
    K,
    representation: PlanarAffine,
    PARTS,
    BM,
    BN,
    BK,
    PAD,
    *,
    capability: Capability,
    output_dtype: DType = DType.BF16,
):
    """Stage only the current BF16 operand tile; codes stay encoded in global."""
    check(representation)
    if not capability.matrix_instructions:
        raise ValueError("the tiled affine contraction requires matrix hardware")
    if representation.coefficient_dtype != DType.BF16 or representation.high_bits:
        raise ValueError("this tiling reads row-addressed BF16 coefficient planes")
    group = representation.group
    N = sum(widths)
    output_at = output_index(M, widths)
    assert K % (PARTS * group) == 0 and (K // PARTS) % BK == 0
    span = K // PARTS

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K // 8), "uint32"),
        C: T.Tensor((N, K // group), "bfloat16"),
        D: T.Tensor((N, K // group), "bfloat16"),
        E: T.Tensor((PARTS, M * N), output_dtype.value),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), PARTS, threads=128) as (bx, by, part):
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
                            global_col = part * span + block * BK + col
                            packed = B[bx * BN + row, global_col // 8]
                            scale = C[bx * BN + row, global_col // group].astype("float32")
                            bias = D[bx * BN + row, global_col // group].astype("float32")
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
            # Fragment stores are opaque to automatic barrier insertion. Result
            # scratch may alias input tiles; all groups must finish reads first.
            T.copy(accum, result)
            T.sync_threads()
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M and bx * BN + j < N:
                    E[
                        part,
                        output_at(by * BM + i, bx * BN + j)
                        if PARTS == 1
                        else (by * BM + i) * N + bx * BN + j,
                    ] = result[i, j].astype("bfloat16")

    return main
