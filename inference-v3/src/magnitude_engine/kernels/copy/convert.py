"""Residency-time conversion of a stored floating parameter into FP32."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.descriptor import WeightTransform


def convert(size: int, dtype: DType, transform: WeightTransform, *, capability: Capability):
    """Widen a stored floating parameter and apply its declared transform.

    This belongs to residency: it happens once when a weight becomes resident,
    and no operation ever selects it.
    """
    cpu = serial(capability)
    @T.macro
    def convert(A, B, i):
        value = A[i].astype("float32")
        B[i] = -T.exp(value) if transform == WeightTransform.NEGATIVE_EXP else value

    @T.prim_func
    def main(A: T.Tensor((size,), dtype.value), B: T.Tensor((size,), "float32")):
        if cpu:
            for i in T.Parallel(size):
                convert(A, B, i)
        else:
            with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
                i = block * 128 + T.get_thread_binding()
                if i < size:
                    convert(A, B, i)

    return main
