"""Which delta-recurrence schedule owns the state of one channel."""

from __future__ import annotations

from dataclasses import dataclass

from magnitude_engine.kernels.recurrence import channel_simd, portable
from magnitude_engine.kernels.semantics import HeadMapping
from magnitude_engine.operations.candidates import Candidate, Plan, Selection
from magnitude_engine.platform.execution import DType


@dataclass(frozen=True)
class RecurrenceShape:
    batch: int
    steps: int
    key_heads: int
    value_heads: int
    key_width: int
    value_width: int
    mapping: HeadMapping
    dtype: DType


def _channel_simd(context, selection: Selection[RecurrenceShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "recurrence.channel_simd",
        (
            context.specialize(
                channel_simd.delta_sequence,
                shape.batch,
                shape.steps,
                shape.key_heads,
                shape.value_heads,
                shape.key_width,
                shape.value_width,
                shape.mapping,
                capability=selection.capability,
                dtype=shape.dtype,
            ),
        ),
    )


def _portable(context, selection: Selection[RecurrenceShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "recurrence.portable",
        (
            context.specialize(
                portable.delta_sequence,
                shape.batch,
                shape.steps,
                shape.key_heads,
                shape.value_heads,
                shape.key_width,
                shape.value_width,
                shape.mapping,
                capability=selection.capability,
                dtype=shape.dtype,
            ),
        ),
    )


def _power_of_two(width: int) -> bool:
    return width > 1 and not width & (width - 1)


TABLE: tuple[Candidate[RecurrenceShape], ...] = (
    Candidate(
        "recurrence.channel_simd",
        lambda s: _power_of_two(s.capability.subgroup_width),
        _channel_simd,
        rank=50,
    ),
    Candidate("recurrence.portable", lambda s: True, _portable, rank=10),
)
