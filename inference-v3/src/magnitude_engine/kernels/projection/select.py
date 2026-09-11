"""Which contraction schedule serves a projection shape.

This is the only module that imports every projection schedule. The predicates
below are the engine's former if-chain rewritten as facts about the four axes:
the representation the weight is resident in, the endpoint capability, the
precision, and the shape. No row names a backend or a container.
"""

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
from magnitude_engine.kernels.projection.planar_affine import (
    finish as planar_finish,
)
from magnitude_engine.kernels.projection.planar_affine import (
    fused_vector,
)
from magnitude_engine.kernels.projection.planar_affine import (
    matrix as planar_matrix,
)
from magnitude_engine.kernels.projection.planar_affine import (
    vector as planar_vector,
)
from magnitude_engine.kernels.projection.planar_affine.layout import partitions
from magnitude_engine.operations.candidates import Candidate, Plan, Scratch, Selection
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.weights.representation import (
    BlockCodec,
    Dense,
    EncodedBlocks,
    HierarchicalAffine,
    HierarchyPacking,
    PlanarAffine,
)


@dataclass(frozen=True)
class ProjectionShape:
    rows: int
    widths: tuple[int, ...]
    """Contiguous output segments. One width is an ordinary linear map."""

    inputs: int
    dtype: DType
    output_dtype: DType

    @property
    def outputs(self) -> int:
        return sum(self.widths)


def _rows(selection: Selection[ProjectionShape]) -> PlanarAffine | None:
    """Planar affine weights whose coefficients are addressed per output row."""
    representation = selection.representation
    if isinstance(representation, PlanarAffine) and representation.coefficient_dtype == DType.BF16:
        return representation
    return None


def _flat(selection: Selection[ProjectionShape]) -> bool:
    """Representations a decoder reads out of one flat operand."""
    representation = selection.representation
    if isinstance(representation, PlanarAffine):
        return representation.coefficient_dtype == DType.F32
    return isinstance(representation, (Dense, EncodedBlocks, HierarchicalAffine))


def _blocked(selection: Selection[ProjectionShape]) -> EncodedBlocks | HierarchicalAffine | None:
    representation = selection.representation
    return (
        representation if isinstance(representation, (EncodedBlocks, HierarchicalAffine)) else None
    )


def _scale_min_hierarchy(selection: Selection[ProjectionShape]) -> bool:
    representation = _blocked(selection)
    return (
        isinstance(representation, HierarchicalAffine)
        and representation.packing == HierarchyPacking.SCALE_MIN_I6
    )


def _groupwise_blocks(selection: Selection[ProjectionShape]) -> bool:
    representation = _blocked(selection)
    return (
        isinstance(representation, HierarchicalAffine)
        and representation.packing == HierarchyPacking.SIGNED_SCALE_I8
    ) or (
        isinstance(representation, EncodedBlocks)
        and representation.codec in (BlockCodec.CODEBOOK_I4, BlockCodec.GROUPED_I8, BlockCodec.F16)
    )


# ---------------------------------------------------------------- planar rows


def _planar_vector_applies(selection: Selection[ProjectionShape]) -> bool:
    shape, capability = selection.shape, selection.capability
    if capability.subgroup_width != 32 or shape.rows >= 8:
        return False
    if _rows(selection) is not None:
        return True
    representation = selection.representation
    return (
        isinstance(representation, PlanarAffine)
        and len(shape.widths) == 1
        and shape.inputs % 512 == 0
    )


def _planar_vector(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "projection.planar_affine.vector",
        (
            context.specialize(
                planar_vector.vector,
                shape.rows,
                shape.widths,
                shape.inputs,
                selection.representation,
                capability=selection.capability,
                dtype=shape.dtype,
                output_dtype=shape.output_dtype,
            ),
        ),
    )


def _planar_matrix_shape(selection: Selection[ProjectionShape]):
    shape = selection.shape
    representation = _rows(selection)
    assert representation is not None
    n = shape.outputs
    parts = partitions(shape.rows, n, shape.inputs)
    large = shape.rows >= 256 and min(n, shape.inputs) >= 512
    bm = 64 if large or shape.rows >= 64 and n >= 8192 else 8 if shape.rows == 1 else 32
    return parts, bm, 64 if n >= 512 else 32, 32, 8 if large else 0


def _planar_matrix_scratch(selection: Selection[ProjectionShape]) -> tuple[Scratch, ...]:
    parts, *_ = _planar_matrix_shape(selection)
    if parts == 1:
        return ()
    shape = selection.shape
    return (
        Scratch(
            "projection.partials",
            TensorSpec((parts, shape.rows * shape.outputs), DType.BF16),
        ),
    )


def _planar_matrix(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    parts, bm, bn, bk, pad = _planar_matrix_shape(selection)
    contraction = context.specialize(
        planar_matrix.matrix,
        shape.rows,
        shape.widths,
        shape.inputs,
        selection.representation,
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
                planar_finish.finish, shape.rows, shape.widths, parts, shape.output_dtype
            ),
        )
    return Plan(
        "projection.planar_affine.matrix",
        executables,
        _planar_matrix_scratch(selection),
        (("partitions", parts),),
    )


def _planar_serial(context, selection: Selection[ProjectionShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "projection.planar_affine.serial",
        (
            context.specialize(
                serial.affine_projection,
                shape.rows,
                shape.widths,
                shape.inputs,
                selection.representation,
                capability=selection.capability,
                output_dtype=shape.output_dtype,
            ),
        ),
    )


# ----------------------------------------------------------------- flat plane


def _matrix_shape(selection: Selection[ProjectionShape]) -> int:
    shape = selection.shape
    # Narrow projections cannot fill the device from M/N tiles alone. Split
    # their contraction axis, then reconcile the partial sums.
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
    parts = _matrix_shape(selection)
    executables = (
        context.specialize(
            matrix.projection,
            shape.rows,
            shape.widths,
            shape.inputs,
            selection.representation,
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
        return Plan(
            name,
            (
                context.specialize(
                    factory,
                    shape.rows,
                    shape.widths,
                    shape.inputs,
                    selection.representation,
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
    return Plan(
        "projection.threadgroup",
        (
            context.specialize(
                threadgroup.projection,
                shape.rows,
                shape.widths,
                shape.inputs,
                selection.representation,
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
    return Plan(
        "projection.serial",
        (
            context.specialize(
                serial.projection,
                shape.rows,
                shape.widths,
                shape.inputs,
                selection.representation,
                row_tile=1,
                dtype=shape.dtype,
                output_dtype=shape.output_dtype,
            ),
        ),
    )


TABLE: tuple[Candidate[ProjectionShape], ...] = (
    Candidate(
        "projection.planar_affine.vector",
        _planar_vector_applies,
        _planar_vector,
        rank=60,
    ),
    Candidate(
        "projection.planar_affine.matrix",
        lambda s: _rows(s) is not None and s.capability.matrix_instructions and s.shape.rows >= 8,
        _planar_matrix,
        rank=50,
        scratch=_planar_matrix_scratch,
    ),
    Candidate(
        "projection.matrix",
        lambda s: _flat(s) and s.capability.matrix_instructions and s.shape.rows >= 8,
        _matrix,
        rank=50,
        scratch=_matrix_scratch,
    ),
    Candidate(
        "projection.hierarchical_scale_min",
        lambda s: (
            _flat(s)
            and _scale_min_hierarchy(s)
            and s.capability.subgroup_width == 32
            and s.shape.inputs % 256 == 0
        ),
        _single(packed_k.projection, "projection.hierarchical_scale_min", output_tile=4),
        rank=40,
    ),
    Candidate(
        "projection.groupwise_blocks",
        lambda s: _flat(s) and _groupwise_blocks(s) and s.capability.subgroup_width == 32,
        _single(blocks.projection, "projection.groupwise_blocks", output_tile=4),
        rank=40,
    ),
    Candidate(
        "projection.subgroup",
        lambda s: _flat(s) and s.capability.subgroup_width > 1,
        _single(subgroup.projection, "projection.subgroup", output_tile=4, pack=8),
        rank=30,
    ),
    Candidate(
        "projection.threadgroup",
        lambda s: _flat(s) and s.capability.threads_per_group > 1,
        _threadgroup,
        rank=20,
    ),
    Candidate(
        "projection.planar_affine.serial",
        lambda s: _rows(s) is not None and s.capability.threads_per_group == 1,
        _planar_serial,
        rank=10,
    ),
    Candidate(
        "projection.serial",
        lambda s: _flat(s) and s.capability.threads_per_group == 1,
        _serial,
        rank=10,
    ),
)


# ------------------------------------------------------------ gated composite


@dataclass(frozen=True)
class GatedShape:
    rows: int
    width: int
    """Each branch's output width; the pair shares it."""

    inputs: int
    dtype: DType
    output_dtype: DType


def _fused_applies(selection: Selection[GatedShape]) -> bool:
    representation = selection.representation
    shape = selection.shape
    return (
        isinstance(representation, PlanarAffine)
        and representation.coefficient_dtype == DType.BF16
        and not representation.high_bits
        and representation.has_bias
        and selection.precision.rounding == Rounding.NATIVE_BF16
        and selection.capability.subgroup_width == 32
        and shape.rows < 8
    )


def _hierarchical_fused_applies(selection: Selection[GatedShape]) -> bool:
    representation = selection.representation
    shape = selection.shape
    return (
        isinstance(representation, HierarchicalAffine)
        and representation.packing == HierarchyPacking.SCALE_MIN_I6
        and selection.precision.rounding == Rounding.NATIVE_BF16
        and selection.capability.subgroup_width == 32
        and shape.rows < 8
        and shape.inputs % representation.supergroup == 0
    )


def _hierarchical_fused(context, selection: Selection[GatedShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "gated.hierarchical_fused_vector",
        (
            context.specialize(
                hierarchical_fused.gated_vector,
                shape.rows,
                shape.width,
                shape.inputs,
                selection.representation,
                capability=selection.capability,
                precision=selection.precision,
            ),
        ),
    )


def _fused(context, selection: Selection[GatedShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "gated.fused_vector",
        (
            context.specialize(
                fused_vector.gated_vector,
                shape.rows,
                shape.width,
                shape.inputs,
                selection.representation,
                capability=selection.capability,
                precision=selection.precision,
            ),
        ),
    )


def _projected_scratch(selection: Selection[GatedShape]) -> tuple[Scratch, ...]:
    shape = selection.shape
    return (
        Scratch("gated.packed", TensorSpec((shape.rows * 2 * shape.width,), shape.output_dtype)),
    )


def _projected(context, selection: Selection[GatedShape]) -> Plan:
    """Two projections into one packed buffer, then the pointwise gate.

    The components are ordinary operations, so this row compiles nothing: the
    plan carries only the scratch the composite needs.
    """
    return Plan("gated.projected", (), _projected_scratch(selection))


GATED_TABLE: tuple[Candidate[GatedShape], ...] = (
    Candidate(
        "gated.hierarchical_fused_vector",
        _hierarchical_fused_applies,
        _hierarchical_fused,
        rank=60,
    ),
    Candidate("gated.fused_vector", _fused_applies, _fused, rank=60),
    Candidate("gated.projected", lambda s: True, _projected, rank=10, scratch=_projected_scratch),
)
