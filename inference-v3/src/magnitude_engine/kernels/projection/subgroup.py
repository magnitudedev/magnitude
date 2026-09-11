"""Subgroup schedule over the common resident-weight interpretation."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.projection.decode import decoder
from magnitude_engine.kernels.projection.layout import output_index, widths_and_outputs
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import Representation, resident_bytes


def projection(
    rows: int,
    widths: int | tuple[int, ...],
    inputs: int,
    representation: Representation,
    *,
    capability: Capability,
    output_tile: int = 4,
    pack: int = 8,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    """One lane reduces a strided slice of a row; the subgroup sums it."""
    logical_widths, outputs = widths_and_outputs(widths)
    output_at = output_index(rows, logical_widths)
    subgroup_width = capability.subgroup_width
    if min(rows, inputs, subgroup_width, output_tile, pack, row_tile) <= 0:
        raise ValueError("projection extents must be positive")
    if subgroup_width <= 1:
        raise ValueError("this schedule requires a subgroup wider than one lane")
    size = resident_bytes(representation, outputs * inputs)
    decode = decoder(representation, outputs * inputs)

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint8"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile),
            T.ceildiv(rows, row_tile),
            threads=output_tile * subgroup_width,
        ) as (block, row_group):
            thread = T.get_thread_binding()
            lane = thread % subgroup_width
            out = block * output_tile + thread // subgroup_width
            partial = T.alloc_local((row_tile,), "float32")
            T.clear(partial)
            for chunk in T.serial(T.ceildiv(inputs, subgroup_width * pack)):
                for j in T.unroll(pack, explicit=True):
                    k = (chunk * pack + j) * subgroup_width + lane
                    if out < outputs and k < inputs:
                        weight = decode(B, out * inputs + k)
                        for r in T.unroll(row_tile, explicit=True):
                            row = row_group * row_tile + r
                            if row < rows:
                                partial[r] += A[row, k].astype("float32") * weight
            for r in T.unroll(row_tile, explicit=True):
                total = T.warp_reduce_sum(partial[r])
                row = row_group * row_tile + r
                if lane == 0 and out < outputs and row < rows:
                    index = output_at(row, out)
                    C[index // outputs, index % outputs] = total

    return main
