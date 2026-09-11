"""Token lookup that leaves the shared table in its resident representation."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.projection.decode import decoder
from magnitude_engine.kernels.projection.planar_affine.layout import check
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    PlanarAffine,
    Representation,
    resident_bytes,
)


def gather(
    rows: int,
    vocabulary: int,
    width: int,
    representation: Representation,
    *,
    capability: Capability,
    threads=128,
    dtype: DType = DType.F32,
):
    """Read selected rows while the shared table stays in its resident form."""
    if min(rows, vocabulary, width, threads) <= 0:
        raise ValueError("invalid encoded embedding geometry")
    decode = decoder(representation, vocabulary * width)
    size = resident_bytes(representation, vocabulary * width)
    cpu = serial(capability)

    @T.prim_func
    def main(
        Indices: T.Tensor((rows,), "int32"),
        W: T.Tensor((size,), "uint8"),
        C: T.Tensor((rows, width), dtype.value),
    ):
        if cpu:
            for row, col in T.Parallel(rows, width):
                if Indices[row] >= 0 and Indices[row] < vocabulary:
                    C[row, col] = decode(W, Indices[row] * width + col)
                else:
                    C[row, col] = T.reinterpret(T.uint32(0x7FC00000), "float32")
        else:
            with T.Kernel(T.ceildiv(width, threads), rows, threads=threads) as (block, row):
                col = block * threads + T.get_thread_binding(0)
                if col < width:
                    if Indices[row] >= 0 and Indices[row] < vocabulary:
                        C[row, col] = decode(W, Indices[row] * width + col)
                    else:
                        C[row, col] = T.reinterpret(T.uint32(0x7FC00000), "float32")

    return main


def planar_gather(N, K, representation: PlanarAffine, *, capability: Capability, ROWS=1):
    """Row-addressed planes gathered without materializing the shared table."""
    check(representation)
    if representation.coefficient_dtype != DType.BF16 or representation.high_bits:
        raise ValueError("this gather reads row-addressed BF16 coefficient planes")
    group = representation.group
    cpu = serial(capability)
    @T.macro
    def load(A, B, C, D, E, row, k):
        if k < K:
            code = (A[D[row], k // 8] >> ((k % 8) * 4)) & T.uint32(15)
            E[row, k] = code.astype("float32") * B[D[row], k // 64].astype("float32") + C[
                D[row], k // 64
            ].astype("float32")

    @T.prim_func
    def main(
        A: T.Tensor((N, K // 8), "uint32"),
        B: T.Tensor((N, K // group), "bfloat16"),
        C: T.Tensor((N, K // group), "bfloat16"),
        D: T.Tensor((ROWS,), "int32"),
        E: T.Tensor((ROWS, K), "bfloat16"),
    ):
        if cpu:
            for row, k in T.Parallel(ROWS, K):
                load(A, B, C, D, E, row, k)
        else:
            with T.Kernel(T.ceildiv(K, 128), ROWS, threads=128) as (block, row):
                k = block * 128 + T.get_thread_binding()
                load(A, B, C, D, E, row, k)

    return main
