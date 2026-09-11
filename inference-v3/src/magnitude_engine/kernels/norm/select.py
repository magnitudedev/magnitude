"""Which row-normalization schedule serves a rounding mode and a subgroup."""

from __future__ import annotations

from dataclasses import dataclass

from magnitude_engine.kernels.norm import portable, subgroup
from magnitude_engine.kernels.precision import Rounding
from magnitude_engine.operations.candidates import Candidate, Plan, Selection
from magnitude_engine.platform.execution import DType


@dataclass(frozen=True)
class NormShape:
    rows: int
    width: int
    epsilon: float
    dtype: DType
    output_dtype: DType


def _subgroup(context, selection: Selection[NormShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "norm.subgroup",
        (
            context.specialize(
                subgroup.norm,
                1,
                shape.width,
                capability=selection.capability,
                precision=selection.precision,
                ROWS=shape.rows,
                epsilon=shape.epsilon,
            ),
        ),
    )


def _portable(context, selection: Selection[NormShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "norm.portable",
        (
            context.specialize(
                portable.rms_norm,
                shape.rows,
                shape.width,
                shape.epsilon,
                capability=selection.capability,
                precision=selection.precision,
                dtype=shape.dtype,
                output_dtype=shape.output_dtype,
            ),
        ),
    )


TABLE: tuple[Candidate[NormShape], ...] = (
    Candidate(
        "norm.subgroup",
        lambda s: (
            s.precision.rounding == Rounding.NATIVE_BF16 and s.capability.subgroup_width == 32
        ),
        _subgroup,
        rank=50,
    ),
    Candidate("norm.portable", lambda s: True, _portable, rank=10),
)
