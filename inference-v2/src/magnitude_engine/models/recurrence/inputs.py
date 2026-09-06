"""Prepared gated-delta inputs shared by recurrence implementations."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class DeltaInputs:
    queries: mx.array
    keys: mx.array
    values: mx.array
    decay: mx.array
    beta: mx.array

    @property
    def length(self) -> int:
        return self.keys.shape[1]

    def prefix(self, count: int) -> DeltaInputs:
        return DeltaInputs(
            *(v[:, :count] for v in (self.queries, self.keys, self.values, self.decay, self.beta))
        )
