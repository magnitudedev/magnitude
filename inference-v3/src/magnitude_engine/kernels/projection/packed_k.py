"""Canonical Q4/Q5 hierarchy contraction.

The schedule knows only the affine parameters and the engine's resident layout.
One lane consumes eight consecutive codes, then applies the shared scale and
bias correction for their 32-element group.
"""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.projection.layout import output_index, widths_and_outputs
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    HierarchicalCoefficients,
    WeightLayout,
    canonical_layout,
    is_scale_min_hierarchy,
)


@T.macro
def _byte(words, offset):
    word = words[offset // 4]
    return ((word >> ((offset % 4) * 8)) & T.uint32(255)).astype("uint32")


@T.macro
def _half(words, offset):
    bits = (_byte(words, offset) | (_byte(words, offset + 1) << 8)).astype("uint16")
    return T.reinterpret(bits, "float16").astype("float32")


@T.macro
def _packed(words, offset, bits, index):
    bit = index * bits
    value = _byte(words, offset + bit // 8) | (_byte(words, offset + bit // 8 + 1) << 8)
    return ((value >> (bit % 8)) & T.uint32((1 << bits) - 1)).astype("float32")


def projection(
    rows: int,
    widths: int | tuple[int, ...],
    inputs: int,
    layout: WeightLayout,
    *,
    capability: Capability,
    output_tile: int = 4,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    logical_widths, outputs = widths_and_outputs(widths)
    output_at = output_index(rows, logical_widths)
    representation = layout.representation
    if capability.subgroup_width != 32:
        raise ValueError("the hierarchy contraction requires a 32-lane subgroup")
    if not is_scale_min_hierarchy(representation):
        raise ValueError("the hierarchy contraction requires canonical Q4_K or Q5_K")
    coefficients = representation.coefficients
    assert isinstance(coefficients, HierarchicalCoefficients)
    if (
        outputs != layout.logical_rows
        or inputs != layout.columns
        or inputs % 256
        or min(rows, output_tile, row_tile) <= 0
        or output_tile % 2
        or layout.nbytes % 4
    ):
        raise ValueError("hierarchy contraction and resident layout differ")

    resident = canonical_layout(representation, layout.elements)
    assert resident.high is None or representation.code.high_bits == 1
    assert resident.biases is not None
    assert resident.super_scale is not None
    assert resident.super_bias is not None
    words = layout.nbytes // 4

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile),
            T.ceildiv(rows, row_tile),
            threads=(output_tile // 2) * 32,
        ) as (block, row_group):
            thread = T.get_thread_binding()
            lane = thread % 32
            first_out = block * output_tile + (thread // 32) * 2
            group = lane // 4
            partial = T.alloc_local((row_tile, 2), "float32")
            dot = T.alloc_local((row_tile,), "float32")
            sums = T.alloc_local((row_tile,), "float32")
            values = T.alloc_local((row_tile, 8), "float32")
            T.clear(partial)
            for chunk in T.serial(inputs // 256):
                for j in T.unroll(8, explicit=True):
                    k = chunk * 256 + lane * 8 + j
                    for r in T.unroll(row_tile, explicit=True):
                        row = row_group * row_tile + r
                        if row < rows:
                            values[r, j] = A[row, k].astype("float32")
                for owned in T.unroll(2, explicit=True):
                    out = first_out + owned
                    if out < outputs:
                        element = (layout.first_row + out) * inputs + chunk * 256
                        base = (element // 256) * resident.tile_bytes
                        low = B[(base + resident.low) // 4 + lane]
                        high = T.uint32(0)
                        if representation.code.high_bits:
                            high = _byte(B, base + resident.high + lane)
                        scale = _packed(B, base + resident.scales, 6, group)
                        bias = _packed(B, base + resident.biases, 6, group)
                        super_scale = _half(B, base + resident.super_scale)
                        super_bias = _half(B, base + resident.super_bias)
                        T.clear(dot)
                        T.clear(sums)
                        for j in T.unroll(8, explicit=True):
                            code = (low >> (j * 4)) & T.uint32(15)
                            if representation.code.high_bits:
                                code = code | (((high >> j) & T.uint32(1)) << 4)
                            for r in T.unroll(row_tile, explicit=True):
                                row = row_group * row_tile + r
                                if row < rows:
                                    value = values[r, j]
                                    dot[r] += value * code.astype("float32")
                                    sums[r] += value
                        for r in T.unroll(row_tile, explicit=True):
                            row = row_group * row_tile + r
                            if row < rows:
                                partial[r, owned] += (
                                    super_scale * scale * dot[r] - super_bias * bias * sums[r]
                                )
            for r in T.unroll(row_tile, explicit=True):
                for owned in T.unroll(2, explicit=True):
                    total = T.warp_reduce_sum(partial[r, owned])
                    row = row_group * row_tile + r
                    out = first_out + owned
                    if lane == 0 and row < rows and out < outputs:
                        index = output_at(row, out)
                        C[index // outputs, index % outputs] = total

    return main
