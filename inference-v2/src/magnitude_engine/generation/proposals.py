"""A proposal has a known width while its token values may still be on device."""

from __future__ import annotations

from typing import cast

import mlx.core as mx


class Proposal:
    def __init__(self, tokens: mx.array):
        if tokens.ndim != 1 or tokens.dtype != mx.int32:
            raise ValueError("proposal tokens must be a one-dimensional int32 array")
        self.tokens = tokens
        self._host: tuple[int, ...] | None = None

    @property
    def count(self) -> int:
        return self.tokens.shape[0]

    @classmethod
    def from_tokens(cls, tokens: tuple[int, ...]) -> Proposal:
        result = cls(mx.array(tokens, dtype=mx.int32))
        result._host = tokens
        return result

    def host(self) -> tuple[int, ...]:
        if self._host is None:
            self._host = tuple(cast(list[int], self.tokens.tolist()))
        return self._host
