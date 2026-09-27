"""Sequence-aligned embedding replacements supplied by a model input adapter."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class EmbeddedInputs:
    """A complete token-aligned embedding stream supplied by a paired target."""

    values: mx.array

    def __post_init__(self):
        if self.values.ndim != 3 or self.values.shape[0] != 1:
            raise ValueError("embedded inputs require [1, tokens, width] values")

    def prefix(self, count):
        return EmbeddedInputs(self.values[:, :count])


@dataclass(frozen=True)
class EmbeddingReplacement:
    start: int
    values: mx.array

    def __post_init__(self):
        if (
            type(self.start) is not int
            or self.start < 0
            or (self.values.ndim != 3 or self.values.shape[0] != 1 or min(self.values.shape) < 1)
        ):
            raise ValueError("embedding replacement requires a start and [1, tokens, width] values")

    @property
    def end(self) -> int:
        return self.start + self.values.shape[1]


def prefix(
    replacements: tuple[EmbeddingReplacement, ...], count: int
) -> tuple[EmbeddingReplacement, ...]:
    return tuple(
        EmbeddingReplacement(row.start, row.values[:, : count - row.start])
        for row in replacements
        if row.start < count
    )


def replace(hidden: mx.array, rows: tuple[tuple[EmbeddingReplacement, ...], ...]) -> mx.array:
    if hidden.ndim != 3 or len(rows) != hidden.shape[0]:
        raise ValueError("embedding replacements must align with decoder rows")
    for index, replacements in enumerate(rows):
        end = 0
        for replacement in replacements:
            if (
                replacement.start < end
                or replacement.end > hidden.shape[1]
                or (replacement.values.shape[2] != hidden.shape[2])
            ):
                raise ValueError("embedding replacements overlap or exceed the decoder geometry")
            hidden[index : index + 1, replacement.start : replacement.end] = (
                replacement.values.astype(hidden.dtype)
            )
            end = replacement.end
    return hidden
