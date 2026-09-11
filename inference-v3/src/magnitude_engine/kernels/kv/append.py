"""Device copies for newly reserved continuation ranges."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.platform.execution import DType


def append_kv(
    rows: int,
    heads: int,
    width: int,
    capacity: int,
    *,
    capability: Capability,
    threads: int = 128,
    dtype: DType = DType.F32,
):
    if min(rows, heads, width, capacity, threads) <= 0 or rows > capacity:
        raise ValueError("invalid KV append geometry")
    cpu = serial(capability)

    @T.macro
    def write(Keys, Values, Offset, KeyStore, ValueStore, index):
        row = index // (heads * width)
        head = index // width % heads
        column = index % width
        destination = Offset[0] + row
        if destination >= 0 and destination < capacity:
            KeyStore[destination, head, column] = Keys[row, head, column]
            ValueStore[destination, head, column] = Values[row, head, column]

    @T.prim_func
    def main(
        Keys: T.Tensor((rows, heads, width), dtype.value),
        Values: T.Tensor((rows, heads, width), dtype.value),
        Offset: T.Tensor((1,), "int32"),
        KeyStore: T.Tensor((capacity, heads, width), dtype.value),
        ValueStore: T.Tensor((capacity, heads, width), dtype.value),
    ):
        if cpu:
            for index in T.Parallel(rows * heads * width):
                write(Keys, Values, Offset, KeyStore, ValueStore, index)
        else:
            with T.Kernel(T.ceildiv(rows * heads * width, threads), threads=threads) as block:
                index = block * threads + T.get_thread_binding(0)
                if index < rows * heads * width:
                    write(Keys, Values, Offset, KeyStore, ValueStore, index)

    return main
