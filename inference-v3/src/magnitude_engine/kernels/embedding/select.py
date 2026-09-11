"""Which gather reads a token row, given the table's resident representation."""

from __future__ import annotations

from dataclasses import dataclass

from magnitude_engine.kernels.embedding import gather
from magnitude_engine.operations.candidates import Candidate, Plan, Selection
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import PlanarAffine


@dataclass(frozen=True)
class EmbeddingShape:
    rows: int
    vocabulary: int
    width: int
    dtype: DType


def _row_planes(selection: Selection[EmbeddingShape]) -> bool:
    representation = selection.representation
    return (
        isinstance(representation, PlanarAffine)
        and representation.coefficient_dtype == DType.BF16
    )


def _planar(context, selection: Selection[EmbeddingShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "embedding.planar_rows",
        (
            context.specialize(
                gather.planar_gather,
                shape.vocabulary,
                shape.width,
                selection.representation,
                capability=selection.capability,
                ROWS=shape.rows,
            ),
        ),
    )


def _gather(context, selection: Selection[EmbeddingShape]) -> Plan:
    shape = selection.shape
    return Plan(
        "embedding.gather",
        (
            context.specialize(
                gather.gather,
                shape.rows,
                shape.vocabulary,
                shape.width,
                selection.representation,
                capability=selection.capability,
                dtype=shape.dtype,
            ),
        ),
    )


TABLE: tuple[Candidate[EmbeddingShape], ...] = (
    Candidate("embedding.planar_rows", _row_planes, _planar, rank=50),
    Candidate("embedding.gather", lambda s: not _row_planes(s), _gather, rank=10),
)
