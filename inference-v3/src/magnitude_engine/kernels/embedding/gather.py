"""Token lookup from one canonical resident allocation."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.projection.decode import decoder
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import WeightLayout


def gather(
    rows: int,
    vocabulary: int,
    width: int,
    layout: WeightLayout,
    *,
    capability: Capability,
    threads: int = 128,
    dtype: DType = DType.F32,
):
    if (
        min(rows, vocabulary, width, threads) <= 0
        or vocabulary != layout.logical_rows
        or width != layout.columns
    ):
        raise ValueError("embedding and resident layout differ")
    decode = decoder(layout.representation, layout.elements)
    size = layout.nbytes
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
                    index = (layout.first_row + Indices[row]) * width + col
                    C[row, col] = decode(W, index)
                else:
                    C[row, col] = T.reinterpret(T.uint32(0x7FC00000), "float32")
        else:
            with T.Kernel(T.ceildiv(width, threads), rows, threads=threads) as (block, row):
                col = block * threads + T.get_thread_binding(0)
                if col < width:
                    if Indices[row] >= 0 and Indices[row] < vocabulary:
                        index = (layout.first_row + Indices[row]) * width + col
                        C[row, col] = decode(W, index)
                    else:
                        C[row, col] = T.reinterpret(T.uint32(0x7FC00000), "float32")

    return main
