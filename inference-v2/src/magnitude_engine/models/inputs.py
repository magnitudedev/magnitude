"""Token-aligned model inputs, including conditioning supplied by a bound drafter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import mlx.core as mx


@dataclass(frozen=True)
class ModelInputs:
    """Arrays stay on device through neural proposal and verification execution.

    Device token arrays are produced by trusted model/generation operations; this
    constructor validates geometry without synchronizing their values to the host.
    External token lists enter through from_tokens, which validates their values.
    Conditioning is aligned as [row, token, ...], so replay slices both together.
    """

    tokens: mx.array
    conditioning: Mapping[str, mx.array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2 or self.tokens.shape[0] != 1 or self.tokens.dtype != mx.int32:
            raise ValueError("sequence model inputs require [1, tokens] int32 IDs")
        for name, value in self.conditioning.items():
            if not name or value.ndim < 2 or value.shape[:2] != self.tokens.shape:
                raise ValueError("model conditioning must align with input tokens")
        object.__setattr__(self, "conditioning", MappingProxyType(dict(self.conditioning)))

    @property
    def count(self) -> int:
        return self.tokens.shape[1]

    @classmethod
    def from_tokens(cls, tokens: tuple[int, ...]) -> ModelInputs:
        if any(type(t) is not int or t < 0 or t > 0x7FFFFFFF for t in tokens):
            raise ValueError("model token IDs must be nonnegative int32 values")
        return cls(mx.array([tokens], dtype=mx.int32))

    def prefix(self, count: int) -> ModelInputs:
        if not 0 <= count <= self.count:
            raise ValueError("model input prefix is outside the input")
        return ModelInputs(
            self.tokens[:, :count],
            {name: value[:, :count] for name, value in self.conditioning.items()},
        )
