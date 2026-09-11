"""Group-wise contractions: payload dot products precede shared coefficients.

A subgroup consumes 256 adjacent logical weights per iteration. Each lane owns
eight coordinates within one quantization group, so it loads that group's scale
once. Resident storage stays in its declared compact representation.
"""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.projection.decode import interpretation
from magnitude_engine.kernels.projection.layout import (
    output_index,
    weight_index,
    widths_and_outputs,
)
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import Affine, Codebook, WeightLayout


def projection(
    rows,
    widths: int | tuple[int, ...],
    inputs,
    layout: WeightLayout,
    *,
    capability: Capability,
    row_tile=1,
    output_tile=4,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    logical_widths, outputs = widths_and_outputs(widths)
    output_at = output_index(rows, logical_widths)
    if capability.subgroup_width != 32:
        raise ValueError("the group contraction consumes 256 weights per 32-lane subgroup")
    representation = layout.representation
    if not isinstance(representation, (Affine, Codebook)):
        raise ValueError("group contraction requires quantized weights")
    if outputs != layout.logical_rows or inputs != layout.columns or layout.nbytes % 2:
        raise ValueError("group contraction and resident layout differ")
    tile = representation.supergroup or representation.group
    if min(rows, inputs, row_tile, output_tile) <= 0 or inputs % tile:
        raise ValueError("invalid group contraction geometry")
    if output_tile % 2:
        raise ValueError("group contraction tiles output rows in pairs")
    at = weight_index(layout)
    parameters, payload = interpretation(representation, layout.elements)
    size = layout.nbytes // 2

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint16"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile),
            T.ceildiv(rows, row_tile),
            threads=32 * (output_tile // 2),
        ) as (block, row_group):
            tid = T.get_thread_binding()
            lane = tid % 32
            first_out = block * output_tile + (tid // 32) * 2
            accum = T.alloc_local((row_tile, 2), "float32")
            dot = T.alloc_local((row_tile,), "float32")
            values = T.alloc_local((row_tile, 8), "float32")
            T.clear(accum)
            for chunk in T.serial(T.ceildiv(inputs, 256)):
                first = chunk * 256 + lane * 8
                if first < inputs:
                    for j in T.unroll(8, explicit=True):
                        k = first + j
                        if k < inputs:
                            for r in T.unroll(row_tile, explicit=True):
                                row = row_group * row_tile + r
                                if row < rows:
                                    values[r, j] = A[row, k].astype("float32")
                    for owned in T.unroll(2, explicit=True):
                        out = first_out + owned
                        T.clear(dot)
                        if out < outputs:
                            storage_first = at(out, first)
                            scale, _ = parameters(B, storage_first)
                            for j in T.unroll(8, explicit=True):
                                k = first + j
                                if k < inputs:
                                    code = payload(B, at(out, k))
                                    for r in T.unroll(row_tile, explicit=True):
                                        row = row_group * row_tile + r
                                        if row < rows:
                                            dot[r] += values[r, j] * code
                            for r in T.unroll(row_tile, explicit=True):
                                accum[r, owned] += dot[r] * scale
            for r in T.unroll(row_tile, explicit=True):
                for owned in T.unroll(2, explicit=True):
                    total = T.warp_reduce_sum(accum[r, owned])
                    row = row_group * row_tile + r
                    out = first_out + owned
                    if lane == 0 and out < outputs and row < rows:
                        index = output_at(row, out)
                        C[index // outputs, index % outputs] = total

    return main
