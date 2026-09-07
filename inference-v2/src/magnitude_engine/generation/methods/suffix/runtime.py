"""Bound generation methods own their per-sequence proposal and observation state."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.generation.proposals import Proposal
from magnitude_engine.models.operations import Task
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.resources.retention import RetainedStorage

from ..contracts import (
    MethodCheckpoint,
    Verification,
)


@dataclass
class SuffixCheckpoint:
    indexed: int
    index: dict[tuple[int, ...], int]
    closed: bool = False

    @property
    def reclaimable(self) -> bool:
        return True

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        from sys import getsizeof

        size = getsizeof(self.index) + sum(
            getsizeof(k) + getsizeof(v) for k, v in self.index.items()
        )
        return (RetainedStorage(self, size),)

    def close(self) -> None:
        self.index.clear()
        self.closed = True


class SuffixSession:
    features: frozenset[str] = frozenset()
    prefill_features: frozenset[str] = frozenset()

    def __init__(self, minimum: int, maximum: int, checkpoint: SuffixCheckpoint | None):
        self.minimum = minimum
        self.maximum = maximum
        self.indexed = 0 if checkpoint is None else checkpoint.indexed
        self.index = {} if checkpoint is None else dict(checkpoint.index)

    def prefill(self, tokens: tuple[int, ...], features: Mapping[str, mx.array]) -> Task[None]:
        yield from ()

    def observe(self, verification: Verification) -> None:
        # Suffix proposals own no neural state to reconcile.
        pass

    def propose(self, context: Sequence[int], limit: int) -> Task[Proposal]:
        yield from ()
        length = len(context)
        if limit <= 0 or length <= self.minimum:
            return Proposal.from_tokens(())
        if self.indexed > length - 1:
            raise ValueError("suffix context moved behind its committed index")
        # Index earlier end positions once, so lookup cost does not grow with context.
        for end in range(self.indexed + 1, length):
            for width in range(self.minimum, min(end, self.maximum) + 1):
                self.index[tuple(context[end - width : end])] = end
        self.indexed = length - 1
        for width in range(min(self.maximum, length - 1), self.minimum - 1, -1):
            end = self.index.get(tuple(context[length - width :]))
            if end is not None:
                return Proposal.from_tokens(tuple(context[end : end + limit]))
        return Proposal.from_tokens(())

    def checkpoint(self) -> SuffixCheckpoint:
        return SuffixCheckpoint(self.indexed, dict(self.index))

    def close(self) -> None:
        self.index.clear()


@component("GENERATION:SPECULATION:MAG:SUFFIX")
class SuffixMethod:
    def __init__(self, minimum: int = 3, maximum: int = 6):
        if minimum < 1 or maximum < minimum:
            raise ValueError("invalid suffix match lengths")
        self.minimum = minimum
        self.maximum = maximum
        self.identity = f"suffix:{minimum}:{maximum}"

    def create(
        self, checkpoint: MethodCheckpoint | None = None, *, target: ModelRuntime[Any, Any]
    ) -> SuffixSession:
        if checkpoint is not None:
            if not isinstance(checkpoint, SuffixCheckpoint) or checkpoint.closed:
                raise ValueError("invalid suffix checkpoint")
        return SuffixSession(self.minimum, self.maximum, checkpoint)
