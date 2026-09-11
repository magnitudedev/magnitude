"""Which attention schedule serves a query batch against a visible history.

Partition counts, head tiles and score storage are properties of particular
schedules, so each candidate owns its own. The operation below this table sees
only how many partials came back, in what dtype, and which scratch they need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from magnitude_engine.kernels.attention import (
    decode_online,
    decode_partitioned,
    materialized,
    merge,
    portable,
    serial,
    streaming,
)
from magnitude_engine.operations.candidates import Candidate, Plan, Scratch, Selection
from magnitude_engine.platform.execution import DType, Executable, TensorSpec


@dataclass(frozen=True)
class HistoryShape:
    capacity: int
    segment_capacity: int
    segments: int


@dataclass(frozen=True)
class AttentionShape:
    rows: int
    heads: int
    kv_heads: int
    width: int
    dtype: DType
    groups: tuple[HistoryShape, ...]

    @property
    def total_capacity(self) -> int:
        return sum(group.segments * group.segment_capacity for group in self.groups)

    @property
    def head_group(self) -> int:
        return self.heads // self.kv_heads


@dataclass(frozen=True)
class AttentionPlan(Plan):
    """Run executables per group, plus the merge that joins their partials."""

    stages: int = 1
    """Executables per group: one run, or the three materialized stages."""

    partitions: tuple[int, ...] = ()
    partial_count: int = 1
    partial_dtype: DType = DType.F32
    score_dtype: DType | None = None
    merge: Executable | None = None


type Partitioner = Callable[[AttentionShape], tuple[int, ...]]
type Stage = Callable[..., tuple[Executable, ...]]


def _partial_count(shape: AttentionShape, partitions: tuple[int, ...]) -> int:
    return sum(
        group.segments * count for group, count in zip(shape.groups, partitions, strict=True)
    )


def _scratch(
    shape: AttentionShape, partitions: tuple[int, ...], score_dtype: DType | None
) -> tuple[Scratch, ...]:
    count = _partial_count(shape, partitions)
    partial_dtype = shape.dtype if count == 1 else DType.F32
    regions = [
        Scratch(
            "attention.statistics", TensorSpec((count, shape.rows, shape.heads, 2), DType.F32)
        )
    ]
    if count > 1:
        regions.append(
            Scratch(
                "attention.partials",
                TensorSpec((count, shape.rows, shape.heads, shape.width), partial_dtype),
            )
        )
    if score_dtype is not None:
        elements = max(
            group.segments * count_per * shape.heads * shape.rows * group.segment_capacity
            for group, count_per in zip(shape.groups, partitions, strict=True)
        )
        regions.append(Scratch("attention.scores", TensorSpec((elements,), score_dtype)))
    return tuple(regions)


def _plan(
    context,
    selection: Selection[AttentionShape],
    name: str,
    stage: Stage,
    partitions: tuple[int, ...],
    *,
    stages: int = 1,
    score_dtype: DType | None = None,
) -> AttentionPlan:
    shape = selection.shape
    count = _partial_count(shape, partitions)
    partial_dtype = shape.dtype if count == 1 else DType.F32
    executables: list[Executable] = []
    for group, per_segment in zip(shape.groups, partitions, strict=True):
        executables.extend(
            stage(context, selection, group, per_segment, partial_dtype, score_dtype)
        )
    merger = (
        context.specialize(
            merge.merge_runs,
            shape.rows,
            shape.heads,
            shape.width,
            count,
            capability=selection.capability,
            output_dtype=shape.dtype,
        )
        if count > 1
        else None
    )
    return AttentionPlan(
        candidate=name,
        executables=tuple(executables),
        scratch=_scratch(shape, partitions, score_dtype),
        detail=tuple(("partitions", per_segment) for per_segment in partitions),
        stages=stages,
        partitions=partitions,
        partial_count=count,
        partial_dtype=partial_dtype,
        score_dtype=score_dtype,
        merge=merger,
    )


# ------------------------------------------------------------------ schedules


def _run(factory) -> Stage:
    def stage(context, selection, group, partitions, partial_dtype, score_dtype):
        shape = selection.shape
        return (
            context.specialize(
                factory,
                shape.rows,
                shape.heads,
                shape.kv_heads,
                shape.width,
                group.capacity,
                segments=group.segments,
                segment_capacity=group.segment_capacity,
                partitions=partitions,
                dtype=shape.dtype,
                output_dtype=partial_dtype,
            ),
        )

    return stage


def _portable_run(context, selection, group, partitions, partial_dtype, score_dtype):
    shape = selection.shape
    return (
        context.specialize(
            portable.run_attention,
            shape.rows,
            shape.heads,
            shape.kv_heads,
            shape.width,
            group.capacity,
            segments=group.segments,
            segment_capacity=group.segment_capacity,
            partitions=partitions,
            head_tile=shape.head_group if shape.rows < 8 else 1,
            dtype=shape.dtype,
            output_dtype=partial_dtype,
        ),
    )


def _serial_run(context, selection, group, partitions, partial_dtype, score_dtype):
    shape = selection.shape
    return (
        context.specialize(
            serial.run_attention,
            shape.rows,
            shape.heads,
            shape.kv_heads,
            shape.width,
            group.capacity,
            segments=group.segments,
            segment_capacity=group.segment_capacity,
            dtype=shape.dtype,
            output_dtype=partial_dtype,
        ),
    )


def _materialized_run(context, selection, group, partitions, partial_dtype, score_dtype):
    """Score, normalize and contract as three dependent stages over one scratch."""
    shape = selection.shape
    count = group.segments * partitions
    arguments = (
        shape.rows,
        shape.heads,
        shape.kv_heads,
        shape.width,
        group.capacity,
        count,
        group.segment_capacity,
    )
    return (
        context.specialize(
            materialized.scores, *arguments, dtype=shape.dtype, score_dtype=score_dtype
        ),
        context.specialize(
            materialized.normalize_bf16 if score_dtype == DType.BF16 else materialized.normalize,
            shape.rows,
            shape.heads,
            count,
            group.segment_capacity,
        ),
        context.specialize(
            materialized.values,
            *arguments,
            dtype=shape.dtype,
            output_dtype=partial_dtype,
            score_dtype=score_dtype,
        ),
    )


# ----------------------------------------------------------------- partitions


def _whole(shape: AttentionShape) -> tuple[int, ...]:
    return tuple(1 for _ in shape.groups)


def _by_span(size: int) -> Partitioner:
    def split(shape: AttentionShape) -> tuple[int, ...]:
        return tuple(
            max(1, (group.segment_capacity + size - 1) // size) for group in shape.groups
        )

    return split


def _online(shape: AttentionShape) -> tuple[int, ...]:
    # A SIMD scans a history span; enough spans to fill the device, rounded to
    # whole subgroup strides.
    span = max(32, ((shape.total_capacity + 64 * 32 - 1) // (64 * 32)) * 32)
    return _by_span(span)(shape)


def _head_parallel(shape: AttentionShape) -> tuple[int, ...]:
    """Small query batches need parallelism along history as well as heads."""
    if shape.rows >= 8:
        return _whole(shape)
    return tuple(
        min(
            (group.segment_capacity + 31) // 32,
            max(1, (group.segment_capacity * shape.head_group + 511) // 512),
        )
        for group in shape.groups
    )


# ----------------------------------------------------------------- candidates


def _subgroup(selection: Selection[AttentionShape]) -> bool:
    return selection.capability.subgroup_width == 32


def _tiled(selection: Selection[AttentionShape]) -> bool:
    return selection.shape.width <= 256 and selection.shape.width % 8 == 0


def _candidate(
    name: str,
    applies: Callable[[Selection[AttentionShape]], bool],
    stage: Stage,
    partitioner: Partitioner,
    rank: int,
    *,
    stages: int = 1,
    score_dtype: DType | None = None,
) -> Candidate[AttentionShape]:
    return Candidate(
        name,
        applies,
        lambda context, selection: _plan(
            context,
            selection,
            name,
            stage,
            partitioner(selection.shape),
            stages=stages,
            score_dtype=score_dtype,
        ),
        rank=rank,
        scratch=lambda selection: _scratch(
            selection.shape, partitioner(selection.shape), score_dtype
        ),
    )


TABLE: tuple[Candidate[AttentionShape], ...] = (
    _candidate(
        "attention.materialized",
        lambda s: (
            s.capability.matrix_instructions
            and _subgroup(s)
            and _tiled(s)
            and s.shape.rows >= 256
            and s.shape.dtype == DType.BF16
            and s.shape.width == 256
        ),
        _materialized_run,
        _whole,
        60,
        stages=3,
        score_dtype=DType.BF16,
    ),
    _candidate(
        "attention.streaming",
        lambda s: (
            s.capability.matrix_instructions and _subgroup(s) and _tiled(s) and s.shape.rows >= 8
        ),
        _run(streaming.run_attention),
        _by_span(4096),
        50,
    ),
    _candidate(
        "attention.decode_online",
        lambda s: (
            _subgroup(s)
            and s.shape.rows < 8
            and s.shape.dtype == DType.BF16
            and s.shape.width == 256
            and s.shape.total_capacity > 8192
            and s.shape.head_group <= 15
        ),
        _run(decode_online.run_attention),
        _online,
        50,
    ),
    _candidate(
        "attention.decode_partitioned",
        lambda s: _subgroup(s) and _tiled(s) and s.shape.rows < 8,
        _run(decode_partitioned.run_attention),
        _by_span(128),
        40,
    ),
    _candidate(
        "attention.portable",
        lambda s: s.capability.matrix_instructions and s.capability.threads_per_group > 1,
        _portable_run,
        _head_parallel,
        20,
    ),
    _candidate(
        "attention.serial",
        lambda s: s.capability.threads_per_group == 1,
        _serial_run,
        _whole,
        10,
    ),
)
