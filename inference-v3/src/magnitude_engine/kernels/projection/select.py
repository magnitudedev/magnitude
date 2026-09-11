"""Projection schedules selected only from numerical parameters and geometry."""

from __future__ import annotations

from dataclasses import dataclass

from magnitude_engine.kernels.precision import Rounding
from magnitude_engine.kernels.projection import (
    blocks,
    hierarchical_fused,
    matrix,
    packed_k,
    serial,
    subgroup,
    threadgroup,
)
from magnitude_engine.kernels.projection.direct_affine import finish as direct_finish
from magnitude_engine.kernels.projection.direct_affine import fused_vector as direct_fused
from magnitude_engine.kernels.projection.direct_affine import matrix as direct_matrix
from magnitude_engine.kernels.projection.direct_affine import vector as direct_vector
from magnitude_engine.kernels.projection.direct_affine.layout import partitions
from magnitude_engine.operations.candidates import Candidate, Plan, Scratch, Selection
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.weights.representation import (
    Affine,
    Codebook,
    CodeInterpretation,
    Dense,
    DirectCoefficients,
    is_scale_min_hierarchy,
)


@dataclass(frozen=True)
class ProjectionShape:
    rows: int
    widths: tuple[int, ...]
    inputs: int
    dtype: DType
    output_dtype: DType

    @property
    def outputs(self) -> int:
        return sum(self.widths)


def _direct(selection: Selection[ProjectionShape]) -> Affine | None:
    representation = selection.representation
    if (
        isinstance(representation, Affine)
        and isinstance(representation.coefficients, DirectCoefficients)
        and representation.code.low_bits == 4
        and not representation.code.high_bits
        and representation.code.interpretation == CodeInterpretation.UNSIGNED
        and representation.coefficients.scale_dtype == DType.BF16
        and representation.coefficients.bias_dtype == DType.BF16
    ):
        return representation
    return None


def _scale_min(selection: Selection) -> bool:
    return is_scale_min_hierarchy(selection.representation)


def _resident(selection: Selection) -> bool:
    return isinstance(selection.representation, (Dense, Affine, Codebook))


def _direct_vector(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    assert selection.layout is not None
    return Plan(
        "projection.direct_affine.vector",
        (
            context.specialize(
                direct_vector.vector,
                shape.rows,
                shape.widths,
                shape.inputs,
                selection.layout,
                capability=selection.capability,
                dtype=shape.dtype,
                output_dtype=shape.output_dtype,
            ),
        ),
    )


def _direct_matrix_shape(selection: Selection[ProjectionShape]):
    shape = selection.shape
    parts = partitions(shape.rows, shape.outputs, shape.inputs)
    large = shape.rows >= 256 and min(shape.outputs, shape.inputs) >= 512
    bm = 64 if large or shape.rows >= 64 and shape.outputs >= 8192 else 8 if shape.rows == 1 else 32
    return parts, bm, 64 if shape.outputs >= 512 else 32, 32, 8 if large else 0


def _direct_matrix_scratch(selection: Selection[ProjectionShape]) -> tuple[Scratch, ...]:
    parts, *_ = _direct_matrix_shape(selection)
    if parts == 1:
        return ()
    shape = selection.shape
    return (
        Scratch(
            "projection.partials",
            TensorSpec((parts, shape.rows * shape.outputs), DType.BF16),
        ),
    )


def _direct_matrix(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    assert selection.layout is not None
    parts, bm, bn, bk, pad = _direct_matrix_shape(selection)
    contraction = context.specialize(
        direct_matrix.matrix,
        shape.rows,
        shape.widths,
        shape.inputs,
        selection.layout,
        parts,
        bm,
        bn,
        bk,
        pad,
        capability=selection.capability,
        output_dtype=DType.BF16 if parts > 1 else shape.output_dtype,
    )
    executables = (contraction,)
    if parts > 1:
        executables += (
            context.specialize(
                direct_finish.finish, shape.rows, shape.widths, parts, shape.output_dtype
            ),
        )
    return Plan(
        "projection.direct_affine.matrix",
        executables,
        _direct_matrix_scratch(selection),
        (("partitions", parts),),
    )


def _matrix_shape(selection: Selection[ProjectionShape]) -> int:
    shape = selection.shape
    tiles = ((shape.rows + 63) // 64) * ((shape.outputs + 63) // 64)
    return min((shape.inputs + 31) // 32, max(1, (256 + tiles - 1) // tiles))


def _matrix_scratch(selection: Selection[ProjectionShape]) -> tuple[Scratch, ...]:
    parts = _matrix_shape(selection)
    if parts == 1:
        return ()
    shape = selection.shape
    return (
        Scratch(
            "projection.partials",
            TensorSpec((parts * shape.rows, shape.outputs), DType.F32),
        ),
    )


def _matrix(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    assert selection.layout is not None
    parts = _matrix_shape(selection)
    executables = (
        context.specialize(
            matrix.projection,
            shape.rows,
            shape.widths,
            shape.inputs,
            selection.layout,
            capability=selection.capability,
            partitions=parts,
            dtype=shape.dtype,
            output_dtype=shape.output_dtype if parts == 1 else DType.F32,
        ),
    )
    if parts > 1:
        executables += (
            context.specialize(
                matrix.merge_partitions,
                shape.rows,
                shape.widths,
                parts,
                output_dtype=shape.output_dtype,
            ),
        )
    return Plan(
        "projection.matrix", executables, _matrix_scratch(selection), (("partitions", parts),)
    )


def _single(factory, name: str, **extra):
    def build(context, selection: Selection[ProjectionShape]) -> Plan:
        shape = selection.shape
        assert selection.layout is not None
        return Plan(
            name,
            (
                context.specialize(
                    factory,
                    shape.rows,
                    shape.widths,
                    shape.inputs,
                    selection.layout,
                    capability=selection.capability,
                    row_tile=1,
                    dtype=shape.dtype,
                    output_dtype=shape.output_dtype,
                    **extra,
                ),
            ),
        )

    return build


def _threadgroup(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    assert selection.layout is not None
    return Plan(
        "projection.threadgroup",
        (
            context.specialize(
                threadgroup.projection,
                shape.rows,
                shape.widths,
                shape.inputs,
                selection.layout,
                output_tile=4,
                reduction_lanes=32,
                row_tile=1,
                dtype=shape.dtype,
                output_dtype=shape.output_dtype,
            ),
        ),
    )


def _serial(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    assert selection.layout is not None
    return Plan(
        "projection.serial",
        (
            context.specialize(
                serial.projection,
                shape.rows,
                shape.widths,
                shape.inputs,
                selection.layout,
                row_tile=1,
                dtype=shape.dtype,
                output_dtype=shape.output_dtype,
            ),
        ),
    )


TABLE: tuple[Candidate[ProjectionShape], ...] = (
    Candidate(
        "projection.direct_affine.vector",
        lambda selection: (
            _direct(selection) is not None
            and selection.capability.subgroup_width == 32
            and selection.shape.rows < 8
        ),
        _direct_vector,
        rank=60,
    ),
    Candidate(
        "projection.direct_affine.matrix",
        lambda selection: (
            _direct(selection) is not None
            and selection.capability.matrix_instructions
            and selection.shape.rows >= 8
        ),
        _direct_matrix,
        rank=50,
        scratch=_direct_matrix_scratch,
    ),
    Candidate(
        "projection.matrix",
        lambda selection: (
            _resident(selection)
            and _direct(selection) is None
            and selection.capability.matrix_instructions
            and selection.shape.rows >= 8
        ),
        _matrix,
        rank=50,
        scratch=_matrix_scratch,
    ),
    Candidate(
        "projection.hierarchical",
        lambda selection: (
            _scale_min(selection)
            and selection.capability.subgroup_width == 32
            and selection.shape.inputs % 256 == 0
        ),
        _single(packed_k.projection, "projection.hierarchical", output_tile=4),
        rank=40,
    ),
    Candidate(
        "projection.groupwise",
        lambda selection: (
            isinstance(selection.representation, (Affine, Codebook))
            and not _scale_min(selection)
            and _direct(selection) is None
            and selection.capability.subgroup_width == 32
        ),
        _single(blocks.projection, "projection.groupwise", output_tile=4),
        rank=40,
    ),
    Candidate(
        "projection.subgroup",
        lambda selection: _resident(selection) and selection.capability.subgroup_width > 1,
        _single(subgroup.projection, "projection.subgroup", output_tile=4, pack=8),
        rank=30,
    ),
    Candidate(
        "projection.threadgroup",
        lambda selection: _resident(selection) and selection.capability.threads_per_group > 1,
        _threadgroup,
        rank=20,
    ),
    Candidate(
        "projection.serial",
        lambda selection: _resident(selection) and selection.capability.threads_per_group == 1,
        _serial,
        rank=10,
    ),
)


@dataclass(frozen=True)
class GatedShape:
    rows: int
    width: int
    inputs: int
    dtype: DType
    output_dtype: DType


def _direct_fused_applies(selection: Selection[GatedShape]) -> bool:
    representation = selection.representation
    coefficients = representation.coefficients if isinstance(representation, Affine) else None
    shape = selection.shape
    return (
        isinstance(representation, Affine)
        and isinstance(coefficients, DirectCoefficients)
        and representation.code.low_bits == 4
        and not representation.code.high_bits
        and representation.code.interpretation == CodeInterpretation.UNSIGNED
        and coefficients.scale_dtype == DType.BF16
        and coefficients.bias_dtype == DType.BF16
        and selection.precision.rounding == Rounding.NATIVE_BF16
        and selection.capability.subgroup_width == 32
        and shape.rows < 8
    )


def _hierarchical_fused_applies(selection: Selection[GatedShape]) -> bool:
    return (
        _scale_min(selection)
        and selection.precision.rounding == Rounding.NATIVE_BF16
        and selection.capability.subgroup_width == 32
        and selection.shape.rows < 8
        and selection.shape.inputs % 256 == 0
    )


def _fused(factory, name: str):
    def build(context, selection: Selection[GatedShape]) -> Plan:
        shape = selection.shape
        assert selection.layout is not None
        return Plan(
            name,
            (
                context.specialize(
                    factory,
                    shape.rows,
                    shape.width,
                    shape.inputs,
                    selection.layout,
                    capability=selection.capability,
                    precision=selection.precision,
                ),
            ),
        )

    return build


def _projected_scratch(selection: Selection[GatedShape]) -> tuple[Scratch, ...]:
    shape = selection.shape
    return (
        Scratch("gated.packed", TensorSpec((shape.rows * 2 * shape.width,), shape.output_dtype)),
    )


def _projected(context, selection: Selection[GatedShape]) -> Plan:
    return Plan("gated.projected", (), _projected_scratch(selection))


GATED_TABLE: tuple[Candidate[GatedShape], ...] = (
    Candidate(
        "gated.hierarchical_fused_vector",
        _hierarchical_fused_applies,
        _fused(hierarchical_fused.gated_vector, "gated.hierarchical_fused_vector"),
        rank=60,
    ),
    Candidate(
        "gated.direct_affine_fused_vector",
        _direct_fused_applies,
        _fused(direct_fused.gated_vector, "gated.direct_affine_fused_vector"),
        rank=60,
    ),
    Candidate(
        "gated.projected", lambda selection: True, _projected, rank=10, scratch=_projected_scratch
    ),
)
