"""Where a planar-affine kernel finds each plane, and how wide its operands are.

Storage is one allocation shared by every planar-affine weight, whatever
container it came from. Kernels whose coefficients are FP32 address that
allocation as one ``uint32`` operand; kernels whose coefficients are BF16 bind
three views of it, because their group stride differs per plane. Both read the
same bytes at the same offsets: ``weights.representation.plane_offsets`` is the
only definition of where a plane starts.
"""

from __future__ import annotations

from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.weights.representation import PlanarAffine, plane_offsets


def check(representation: object) -> PlanarAffine:
    if not isinstance(representation, PlanarAffine):
        raise TypeError("this schedule reads a planar affine representation")
    return representation


def operand_specs(
    representation: PlanarAffine, outputs: int, inputs: int
) -> tuple[TensorSpec, ...]:
    """The weight operands a schedule for this representation binds.

    FP32 coefficients are read out of the flat word plane, so one operand spans
    the whole allocation. BF16 coefficients are read as rows, so the three
    planes are bound as three views of it.
    """
    check(representation)
    elements = outputs * inputs
    offsets = plane_offsets(representation, elements)
    if representation.coefficient_dtype == DType.F32:
        return (TensorSpec((offsets.words,), DType.U32),)
    if representation.high_bits or not representation.has_bias:
        raise ValueError("row-addressed planes carry a low plane, scales and biases")
    group = representation.group
    return (
        TensorSpec((outputs, inputs * representation.bits // 32), DType.U32),
        TensorSpec((outputs, inputs // group), representation.coefficient_dtype),
        TensorSpec((outputs, inputs // group), representation.coefficient_dtype),
    )


def plane_byte_offsets(representation: PlanarAffine, elements: int) -> tuple[int, ...]:
    """Byte offsets matching ``operand_specs``, within the one allocation."""
    offsets = plane_offsets(representation, elements)
    if representation.coefficient_dtype == DType.F32:
        return (0,)
    assert offsets.biases is not None
    return (offsets.low * 4, offsets.scales * 4, offsets.biases * 4)


def partitions(m, n, k):
    parts = min(max(1, 512 // (((m + 31) // 32) * ((n + 31) // 32))), k // 64)
    while k % (parts * 64):
        parts -= 1
    return parts
