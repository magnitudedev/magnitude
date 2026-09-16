"""Static, portable contraction strategies; no target identity or instruction facts."""

from dataclasses import dataclass
from enum import StrEnum


class OperandPreparation(StrEnum):
    WHOLE = "whole"
    SLICED = "sliced"
    SHARED = "shared"
    FACTORED = "factored"


@dataclass(frozen=True, slots=True)
class AffineSchedule:
    """Storage and lifetime choices under one decoded-operand numerical contract.

    Decoding, coefficient arithmetic and accumulation retain FP32 precision.
    Slicing may change reduction order, never introduce operand rounding.
    The factored strategy distributes affine coefficients over each group dot
    product, keeping exact low-bit codes in the native activation dtype.
    """

    preparation: OperandPreparation
    slice_k: int = 0

    def __post_init__(self):
        if not isinstance(self.preparation, OperandPreparation):
            raise TypeError("operand preparation must be a portable strategy")
        if self.preparation == OperandPreparation.SLICED:
            if self.slice_k <= 0:
                raise ValueError("sliced contraction needs a positive static K extent")
        elif self.slice_k != 0:
            raise ValueError("only sliced preparation has a slice extent")

    def reduction_extent(self, bk: int) -> int:
        step = self.slice_k if self.preparation == OperandPreparation.SLICED else bk
        if bk <= 0 or bk % step:
            raise ValueError("contraction slices must exactly cover the physical K tile")
        return step


def affine_schedule_family(bk: int) -> tuple[AffineSchedule, ...]:
    """A bounded vocabulary whose legality is decided by TileLang compilation."""
    return (
        AffineSchedule(OperandPreparation.WHOLE),
        *(
            AffineSchedule(OperandPreparation.SLICED, step)
            for step in (8, 16, 32)
            if step <= bk and bk % step == 0
        ),
        AffineSchedule(OperandPreparation.SHARED),
        AffineSchedule(OperandPreparation.FACTORED),
    )


@dataclass(frozen=True, slots=True)
class MatrixTile:
    rows: int
    columns: int
    reduction: int
    threads: int


@dataclass(frozen=True, slots=True)
class AffineTile(MatrixTile):
    operands: AffineSchedule


def select_affine_tile(context, source, weights, tile, *, template, name, workload=()):
    """Bounded physical alternatives for a single decoded-operand template."""
    from ..compiler.schedules import select_schedule
    from .packed import affine_shared_bytes

    bm, bn, bk, threads = tile
    default = AffineTile(bm, bn, bk, threads, AffineSchedule(OperandPreparation.WHOLE))
    candidates = tuple(
        AffineTile(rows, bn, bk, threads, operands)
        for rows in sorted({8, 16, 32, 64, bm})
        for operands in affine_schedule_family(bk)
        if affine_shared_bytes(rows, bn, bk, source.dtype, *weights, schedule=operands)
        <= context.compiler_target.shared_memory_bytes
    )
    return select_schedule(
        context,
        name,
        candidates,
        default,
        template=template,
        workload=(source, tuple(weights), workload),
    )


@dataclass(frozen=True, slots=True)
class AffineRegionTile(AffineTile):
    output_columns: int


def select_affine_region(context, source, contractions, tile, *, template, name, workload):
    """Select a common traversal for a region with several contraction widths.

    Each contraction is (is output projection, K extent, weight specifications).
    Resource qualification covers the peak of *all* stages before selecting a
    candidate; one stage cannot borrow another stage's smaller shared footprint.
    """
    from ..compiler.schedules import select_schedule
    from .packed import affine_shared_bytes

    bm, bn, bk, threads = tile
    default = AffineRegionTile(
        bm, bn, bk, threads, AffineSchedule(OperandPreparation.WHOLE), 2 * bn
    )
    candidates = tuple(
        AffineRegionTile(rows, bn, bk, threads, operands, output_columns)
        for rows in sorted({8, 16, 32, 64, bm})
        for output_columns in (bn, 2 * bn)
        for operands in affine_schedule_family(bk)
        if all(
            (not operands.slice_k or reduction % operands.slice_k == 0)
            and affine_shared_bytes(
                rows,
                output_columns if is_output else 2 * bn,
                reduction,
                source.dtype,
                *weights,
                schedule=operands,
            )
            <= context.compiler_target.shared_memory_bytes
            for is_output, reduction, weights in contractions
        )
    )
    return select_schedule(
        context,
        name,
        candidates,
        default,
        template=template,
        workload=(source, contractions, workload),
    )


class StateAccumulation(StrEnum):
    FRAGMENT_UPDATE = "fragment-update"
    SHARED_UPDATE = "shared-update"


@dataclass(frozen=True, slots=True)
class RecurrentSchedule:
    chunk: int
    columns: int
    threads: int
    state_accumulation: StateAccumulation


class ProbabilityTransfer(StrEnum):
    SHARED = "shared"
    INFERRED = "inferred-fragment"
