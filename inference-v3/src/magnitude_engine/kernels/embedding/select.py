"""The canonical resident gather for a token table."""

from __future__ import annotations

from dataclasses import dataclass

from magnitude_engine.kernels.embedding import gather
from magnitude_engine.operations.candidates import Candidate, Plan, Selection
from magnitude_engine.platform.execution import DType


@dataclass(frozen=True)
class EmbeddingShape:
    rows: int
    vocabulary: int
    width: int
    dtype: DType


def _gather(context, selection: Selection[EmbeddingShape]) -> Plan:
    shape = selection.shape
    assert selection.layout is not None
    return Plan(
        "embedding.gather",
        (
            context.specialize(
                gather.gather,
                shape.rows,
                shape.vocabulary,
                shape.width,
                selection.layout,
                capability=selection.capability,
                dtype=shape.dtype,
            ),
        ),
    )


TABLE: tuple[Candidate[EmbeddingShape], ...] = (
    Candidate("embedding.gather", lambda selection: selection.layout is not None, _gather),
)
