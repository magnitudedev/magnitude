"""Pointwise combinations of two operands, in either rounding mode."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.precision import Precision, Rounding, native_sigmoid
from magnitude_engine.kernels.semantics import Pointwise
from magnitude_engine.platform.execution import DType


def pointwise(
    size: int,
    kind: Pointwise,
    *,
    capability: Capability,
    precision: Precision,
    threads: int = 128,
    dtype: DType = DType.F32,
    second_dtype: DType | None = None,
    output_dtype: DType | None = None,
):
    second_dtype = dtype if second_dtype is None else second_dtype
    output_dtype = dtype if output_dtype is None else output_dtype
    if min(size, threads) <= 0 or not isinstance(kind, Pointwise):
        raise ValueError("invalid pointwise program")
    cpu = serial(capability)
    native_rounding = precision.rounding == Rounding.NATIVE_BF16

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
