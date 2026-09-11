"""Dense operand copy used when assembling model inputs on the endpoint."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.platform.execution import DType


def copy(
    size: int,
    *,
    capability: Capability,
    threads: int = 128,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    cpu = serial(capability)

    @T.prim_func
    def main(A: T.Tensor((size,), dtype.value), B: T.Tensor((size,), output_dtype.value)):
        if cpu:
            for index in T.Parallel(size):
                B[index] = A[index]
        else:
            with T.Kernel(T.ceildiv(size, threads), threads=threads) as block:
                index = block * threads + T.get_thread_binding(0)
                if index < size:
                    B[index] = A[index]

    return main
