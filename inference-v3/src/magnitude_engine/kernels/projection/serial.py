"""Host schedule for an encoded contraction; no threadgroup or fragment semantics."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.projection.decode import decoder
from magnitude_engine.kernels.projection.planar_affine.layout import (
    check,
    output_index,
)
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    PlanarAffine,
    Representation,
    resident_bytes,
)


def projection(
    rows: int,
    outputs: int,
    inputs: int,
    representation: Representation,
    *,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    """The same contraction where the endpoint has no threadgroups."""
    if min(rows, outputs, inputs, row_tile) <= 0:
        raise ValueError("invalid encoded projection geometry")
    decode = decoder(representation, outputs * inputs)
    size = resident_bytes(representation, outputs * inputs)

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint8"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        for row_group, out in T.Parallel(T.ceildiv(rows, row_tile), outputs):
            total = T.alloc_local((row_tile,), "float32")
            for r in T.unroll(row_tile, explicit=True):
                total[r] = 0
            for k in T.serial(inputs):
                weight = decode(B, out * inputs + k)
                for r in T.unroll(row_tile, explicit=True):
                    row = row_group * row_tile + r
                    if row < rows:
                        total[r] += A[row, k].astype("float32") * weight
            for r in T.unroll(row_tile, explicit=True):
                row = row_group * row_tile + r
                if row < rows:
                    C[row, out] = total[r]

    return main


def affine_projection(
    M,
    widths: tuple[int, ...],
    K,
    representation: PlanarAffine,
    *,
    capability: Capability,
    output_dtype: DType,
):
    """The row-addressed affine contraction where a group is one work item."""
    check(representation)
    if representation.coefficient_dtype != DType.BF16 or representation.high_bits:
        raise ValueError("this schedule reads row-addressed BF16 coefficient planes")
    group = representation.group
    cpu = serial(capability)
    N = sum(widths)
    output_at = output_index(M, widths)

    @T.macro
    def dot(A, B, C, D, E, row, col):
        total = T.alloc_local((1,), "float32")
        total[0] = 0
        for k in T.serial(K):
            code = (B[col, k // 8] >> ((k % 8) * 4)) & T.uint32(15)
            weight = (
                code.astype("float32") * C[col, k // group].astype("float32")
                + D[col, k // group].astype("float32")
            ).astype("bfloat16")
            total[0] += A[row, k].astype("float32") * weight.astype("float32")
        E[output_at(row, col)] = total[0].astype("bfloat16")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K // 8), "uint32"),
        C: T.Tensor((N, K // group), "bfloat16"),
        D: T.Tensor((N, K // group), "bfloat16"),
        E: T.Tensor((M * N,), output_dtype.value),
    ):
        if cpu:
            for row, col in T.Parallel(M, N):
                dot(A, B, C, D, E, row, col)
        else:
            with T.Kernel(T.ceildiv(N, 128), M, threads=128) as (block, row):
                col = block * 128 + T.get_thread_binding()
                if col < N:
                    dot(A, B, C, D, E, row, col)

    return main
