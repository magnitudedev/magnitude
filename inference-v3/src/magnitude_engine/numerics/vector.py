"""Pointwise and row normalization programs with CPU and GPU schedules."""

import math

import tilelang.language as T

from magnitude_engine.numerics.semantics import Pointwise
from magnitude_engine.platform.execution import DType


def pointwise(
    size: int,
    kind: Pointwise,
    *,
    cpu: bool,
    threads: int = 128,
    dtype: DType = DType.F32,
    second_dtype: DType | None = None,
    output_dtype: DType | None = None,
    native_rounding: bool = False,
):
    second_dtype = dtype if second_dtype is None else second_dtype
    output_dtype = dtype if output_dtype is None else output_dtype
    if min(size, threads) <= 0 or not isinstance(kind, Pointwise):
        raise ValueError("invalid pointwise program")

    from magnitude_engine.numerics.native_bf16 import native_sigmoid

    @T.macro
    def apply(A, B, C, i):
        a, b = A[i].astype("float32"), B[i].astype("float32")
        if kind == Pointwise.ADD:
            C[i] = a + b
        elif kind == Pointwise.MULTIPLY:
            C[i] = a * b
        elif kind == Pointwise.SILU_PRODUCT:
            C[i] = (
                (a * native_sigmoid(a).astype("float32")).astype("bfloat16").astype("float32") * b
                if native_rounding
                else (a / (1 + T.exp(-a))) * b
            )
        else:
            C[i] = (
                a * native_sigmoid(b).astype("float32") if native_rounding else a / (1 + T.exp(-b))
            )

    @T.prim_func
    def main(
        A: T.Tensor((size,), dtype.value),
        B: T.Tensor((size,), second_dtype.value),
        C: T.Tensor((size,), output_dtype.value),
    ):
        if cpu:
            for i in T.Parallel(size):
                apply(A, B, C, i)
        else:
            with T.Kernel(T.ceildiv(size, threads), threads=threads) as block:
                i = block * threads + T.get_thread_binding(0)
                if i < size:
                    apply(A, B, C, i)

    return main


def rms_norm(
    rows: int,
    width: int,
    epsilon: float,
    *,
    cpu: bool,
    threads: int = 128,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
    native_rounding: bool = False,
):
    """Apply multiplicative norm weights; artifact-specific offsets are resolved at binding."""
    if min(rows, width, threads) <= 0 or threads & (threads - 1):
        raise ValueError("invalid normalization geometry")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization epsilon must be positive and finite")

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
