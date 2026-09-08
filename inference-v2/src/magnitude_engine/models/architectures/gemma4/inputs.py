"""Gemma's projected image operands, vocabulary mapping, and local visibility."""

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.models.embeddings.replacement import EmbeddingReplacement, prefix
from magnitude_engine.models.inputs import ModelInputs


@dataclass(frozen=True)
class GemmaInputs:
    embeddings: tuple[EmbeddingReplacement, ...]
    language: mx.array
    key_ends: mx.array | None = None

    def __post_init__(self):
        if (
            self.language.ndim != 2
            or self.language.shape[0] != 1
            or self.language.dtype != mx.bool_
            or (
                self.key_ends is not None
                and (self.key_ends.shape != self.language.shape or self.key_ends.dtype != mx.int32)
            )
        ):
            raise ValueError(
                "Gemma inputs require aligned language masks and local attention bounds"
            )

    def prefix(self, count: int) -> "GemmaInputs":
        return GemmaInputs(
            prefix(self.embeddings, count),
            self.language[:, :count],
            None if self.key_ends is None else self.key_ends[:, :count],
        )


def batch_key_ends(inputs: tuple[ModelInputs, ...], offsets: tuple[int, ...]) -> mx.array | None:
    """Causal rows need no override; mixed batches preserve each row's visibility."""
    ends = tuple(row.data.key_ends if isinstance(row.data, GemmaInputs) else None for row in inputs)
    if all(end is None for end in ends):
        return None
    return mx.concatenate(
        [
            end if end is not None else mx.arange(row.count, dtype=mx.int32)[None] + offset + 1
            for row, offset, end in zip(inputs, offsets, ends, strict=True)
        ]
    )
