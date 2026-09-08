"""Qwen decoder operands: rotary coordinates and projected embedding replacements."""

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.models.embeddings.replacement import EmbeddingReplacement, prefix
from magnitude_engine.models.inputs import ModelInputs


@dataclass(frozen=True)
class QwenInputs:
    positions: mx.array
    embeddings: tuple[EmbeddingReplacement, ...] = ()

    def __post_init__(self):
        if self.positions.dtype != mx.int32 or not (
            self.positions.shape == (1,)
            or (self.positions.ndim == 3 and self.positions.shape[:2] == (3, 1))
        ):
            raise ValueError("Qwen inputs require a row offset or [3, 1, tokens] coordinates")

    def prefix(self, count: int) -> "QwenInputs":
        return QwenInputs(
            self.positions if self.positions.ndim == 1 else self.positions[:, :, :count],
            prefix(self.embeddings, count),
        )


def batch_positions(inputs: tuple[ModelInputs, ...], offsets: tuple[int, ...]) -> mx.array | None:
    if all(row.data is None for row in inputs):
        return None
    if any(row.data is not None and not isinstance(row.data, QwenInputs) for row in inputs):
        raise ValueError("Qwen decoder received incompatible model input data")
    coordinates = tuple(
        row.data.positions if isinstance(row.data, QwenInputs) else mx.array([offset], mx.int32)
        for row, offset in zip(inputs, offsets, strict=True)
    )
    if all(value.ndim == 1 for value in coordinates):
        return coordinates[0] if len(coordinates) == 1 else mx.concatenate(list(coordinates))
    width = inputs[0].count
    expanded = tuple(
        mx.broadcast_to(value[:, None] + mx.arange(width, dtype=mx.int32), (3, 1, width))
        if value.ndim == 1
        else value
        for value in coordinates
    )
    if any(value.shape != (3, 1, width) for value in expanded):
        raise ValueError("Qwen coordinates must align with decoder input width")
    return expanded[0] if len(expanded) == 1 else mx.concatenate(list(expanded), axis=1)
