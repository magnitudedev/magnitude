"""Row normalization with the artifact's multiplicative weights."""

import math

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.precision import Precision, Rounding
from magnitude_engine.platform.execution import DType


def rms_norm(
    rows: int,
    width: int,
    epsilon: float,
    *,
    capability: Capability,
    precision: Precision,
    threads: int = 128,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    """Apply multiplicative norm weights; artifact-specific offsets are resolved at binding."""
    if min(rows, width, threads) <= 0 or threads & (threads - 1):
        raise ValueError("invalid normalization geometry")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization epsilon must be positive and finite")
    cpu = serial(capability)
    native_rounding = precision.rounding == Rounding.NATIVE_BF16

    @T.prim_func
    def main(
        A: T.Tensor((rows, width), dtype.value),
        W: T.Tensor((width,), "float32"),
        C: T.Tensor((rows, width), output_dtype.value),
    ):
        if cpu:
            for row in T.Parallel(rows):
                total = T.alloc_local((1,), "float32")
                total[0] = 0
                for col in T.serial(width):
                    total[0] += A[row, col].astype("float32") * A[row, col].astype("float32")
                inverse = T.rsqrt(total[0] / width + epsilon)
                for col in T.serial(width):
                    C[row, col] = (
                        (A[row, col].astype("float32") * inverse)
                        .astype(dtype.value)
                        .astype("float32")
                        * W[col]
                        if native_rounding
                        else A[row, col].astype("float32") * inverse * W[col]
                    )
        else:
            with T.Kernel(rows, threads=threads) as row:
                lane = T.get_thread_binding(0)
                total = T.alloc_local((1,), "float32")
                shared = T.alloc_shared((threads,), "float32")
                total[0] = 0
                for chunk in T.serial(T.ceildiv(width, threads)):
                    col = chunk * threads + lane
                    if col < width:
                        total[0] += A[row, col].astype("float32") * A[row, col].astype("float32")
                shared[lane] = total[0]
                T.sync_threads()
                for step in T.unroll(int(math.log2(threads))):
                    if lane < (threads >> (step + 1)):
                        shared[lane] += shared[lane + (threads >> (step + 1))]
                    T.sync_threads()
                inverse = T.rsqrt(shared[0] / width + epsilon)
                for chunk in T.serial(T.ceildiv(width, threads)):
                    col = chunk * threads + lane
                    if col < width:
                        C[row, col] = (
                            (A[row, col].astype("float32") * inverse)
                            .astype(dtype.value)
                            .astype("float32")
                            * W[col]
                            if native_rounding
                            else A[row, col].astype("float32") * inverse * W[col]
                        )

    return main
