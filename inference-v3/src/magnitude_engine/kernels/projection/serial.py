"""Host schedule for an encoded contraction; no threadgroup or fragment semantics."""

import tilelang.language as T

from magnitude_engine.kernels.projection.decode import decoder
from magnitude_engine.kernels.projection.layout import (
    output_index,
    weight_index,
    widths_and_outputs,
)
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import WeightLayout


def projection(
    rows: int,
    widths: int | tuple[int, ...],
    inputs: int,
    layout: WeightLayout,
    *,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    """The same contraction where the endpoint has no threadgroups."""
    logical_widths, outputs = widths_and_outputs(widths)
    if min(rows, inputs, row_tile) <= 0:
        raise ValueError("invalid encoded projection geometry")
    if outputs != layout.logical_rows or inputs != layout.columns:
        raise ValueError("projection and resident layout differ")
    output_at = output_index(rows, logical_widths)
    at = weight_index(layout)
    decode = decoder(layout.representation, layout.elements)
    size = layout.nbytes

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
                weight = decode(B, at(out, k))
                for r in T.unroll(row_tile, explicit=True):
                    row = row_group * row_tile + r
                    if row < rows:
                        total[r] += A[row, k].astype("float32") * weight
            for r in T.unroll(row_tile, explicit=True):
                row = row_group * row_tile + r
                if row < rows:
                    index = output_at(row, out)
                    C[index // outputs, index % outputs] = total[r]

    return main
