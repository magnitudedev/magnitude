"""A portable fused GEMV over any resident representation a decoder can read.

Reduction lanes and the output tile are candidate parameters, not device facts;
the schedule stages its partial sums through threadgroup memory, so it needs a
group but no subgroup.
"""

import math

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
    output_tile: int = 4,
    reduction_lanes: int = 32,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    """A portable fused GEMV baseline, also valid for small batched inputs.

    Prefill matrix-tiled candidates will share the decoder and operation contract.
    Reduction lanes and output tile are candidate parameters, not device facts.
    """
    logical_widths, outputs = widths_and_outputs(widths)
    output_at = output_index(rows, logical_widths)
    if min(rows, inputs, output_tile, reduction_lanes, row_tile) <= 0:
        raise ValueError("projection extents and tile sizes must be positive")
    if reduction_lanes & (reduction_lanes - 1):
        raise ValueError("reduction lane count must be a power of two")
    if outputs != layout.logical_rows or inputs != layout.columns:
        raise ValueError("projection and resident layout differ")
    at = weight_index(layout)
    decode = decoder(layout.representation, layout.elements)
    size = layout.nbytes

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint8"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile),
            T.ceildiv(rows, row_tile),
            threads=output_tile * reduction_lanes,
        ) as (block, row_group):
            partial = T.alloc_fragment((row_tile, output_tile, reduction_lanes), "float32")
            shared = T.alloc_shared((row_tile, output_tile, reduction_lanes), "float32")
            T.clear(partial)
            for chunk in T.serial(T.ceildiv(inputs, reduction_lanes)):
                for i, j in T.Parallel(output_tile, reduction_lanes):
                    out = block * output_tile + i
                    k = chunk * reduction_lanes + j
                    if out < outputs and k < inputs:
                        weight = decode(B, at(out, k))
                        for r in T.unroll(row_tile, explicit=True):
                            row = row_group * row_tile + r
                            if row < rows:
                                partial[r, i, j] += A[row, k].astype("float32") * weight
            for r, i, j in T.Parallel(row_tile, output_tile, reduction_lanes):
                shared[r, i, j] = partial[r, i, j]
            T.sync_threads()
            for step in T.unroll(int(math.log2(reduction_lanes))):
                for r, i, j in T.Parallel(row_tile, output_tile, reduction_lanes):
                    if j < (reduction_lanes >> (step + 1)):
                        shared[r, i, j] += shared[r, i, j + (reduction_lanes >> (step + 1))]
                T.sync_threads()
            for r, i in T.Parallel(row_tile, output_tile):
                row = row_group * row_tile + r
                if block * output_tile + i < outputs and row < rows:
                    out = block * output_tile + i
                    index = output_at(row, out)
                    C[index // outputs, index % outputs] = shared[r, i, 0]

    return main
