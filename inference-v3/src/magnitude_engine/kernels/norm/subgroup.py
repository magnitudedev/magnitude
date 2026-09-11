"""The NATIVE_BF16 subgroup normalization, rounding at every boundary.

Its applicability — this rounding mode and a 32-lane subgroup — is a candidate
predicate, not a separate operation.
"""

import struct

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.precision import Precision, Rounding


def norm(
    H,
    D,
    *,
    capability: Capability,
    precision: Precision,
    gain=1.0,
    ROWS=1,
    STRIDE=None,
    OFFSET=0,
    epsilon=1e-6,
):
    if capability.subgroup_width != 32:
        raise ValueError("this normalization reduces across a 32-lane subgroup")
    if precision.rounding != Rounding.NATIVE_BF16:
        raise ValueError("this normalization rounds the way NATIVE_BF16 does")
    STRIDE = H * D if STRIDE is None else STRIDE
    threads = min(D // 4, 1024)
    gain_bits = struct.unpack("I", struct.pack("f", gain))[0]

    @T.prim_func
    def main(
        A: T.Tensor((ROWS, STRIDE), "bfloat16"),
        B: T.Tensor((D,), "float32"),
        C: T.Tensor((ROWS, H, D), "bfloat16"),
    ):
        with T.Kernel(H, ROWS, threads=threads) as (h, row):
            lane = T.get_thread_binding()
            partial = T.alloc_shared((32,), "float32")
            squares = T.alloc_local((1,), "float32")
            squares[0] = 0
            for chunk in T.serial(T.ceildiv(D, threads * 4)):
                for j in T.serial(4):
                    d = chunk * threads * 4 + lane * 4 + j
                    if d < D:
                        x = A[row, OFFSET + h * D + d].astype("float32")
                        squares[0] += x * x
            if lane < 32:
                partial[lane] = 0
            T.sync_threads()
            subtotal = T.warp_reduce_sum(squares[0])
            if lane % 32 == 0:
                partial[lane // 32] = subtotal
            T.sync_threads()
            if lane < 32:
                total = T.warp_reduce_sum(partial[lane])
                if lane == 0:
                    partial[0] = total
            T.sync_threads()
            inverse = T.rsqrt(partial[0] / D + epsilon)
            for chunk in T.serial(T.ceildiv(D, threads * 4)):
                for j in T.serial(4):
                    d = chunk * threads * 4 + lane * 4 + j
                    if d < D:
                        scaled = (
                            (A[row, OFFSET + h * D + d].astype("float32") * inverse)
                            .astype("bfloat16")
                            .astype("float32")
                        )
                        weighted = (
                            (scaled * B[d].astype("float32")).astype("bfloat16").astype("float32")
                        )
                        C[row, h, d] = weighted * T.reinterpret(T.uint32(gain_bits), "float32")

    return main
