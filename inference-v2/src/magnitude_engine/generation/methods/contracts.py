"""Bound generation methods own their per-sequence proposal and observation state."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx

from magnitude_engine.models.operations import Task
from magnitude_engine.models.runtime import ModelRuntime, ModelSequence
from magnitude_engine.resources.retention import RetainedStorage

from ..proposals import Proposal
from ..sampling import SequenceSampler


class MethodCheckpoint(Protocol):
    @property
    def reclaimable(self) -> bool: ...

    def close(self) -> None: ...
    def retained_storage(self) -> tuple[RetainedStorage, ...]: ...


@dataclass(frozen=True)
class Verification:
    """Target execution followed by publication of its next token.

    Proposed inputs may be accepted partially. An externally determined block (such
    as grammar-forced tokens) is fully committed without calling propose first; the
    method must observe its target features just as it observes ordinary advancement.
    """

    inputs: tuple[int, ...]
    accepted_inputs: int
    next_token: int
    features: Mapping[str, mx.array]


class MethodSession(Protocol):
    features: frozenset[str]
    prefill_features: frozenset[str]

    def prefill(self, tokens: tuple[int, ...], features: Mapping[str, mx.array]) -> None: ...
    def propose(self, context: Sequence[int], limit: int) -> Task[Proposal]: ...
    def observe(self, verification: Verification) -> None: ...
    def checkpoint(self) -> MethodCheckpoint: ...
    def close(self) -> None: ...


class GenerationMethod(Protocol):
    identity: str

    def create(
        self, checkpoint: MethodCheckpoint | None = None, *, target: ModelRuntime[Any, Any]
    ) -> MethodSession: ...


@dataclass(frozen=True)
class CausalResult:
    tokens: tuple[int, ...]
    evaluated_inputs: int


@runtime_checkable
class CausalSession(Protocol):
    """An ordinary method can advance a bounded span using device token feedback.

    The caller supplies a history-independent sampler and an output allowance.
    All issued model work must complete before return. Every evaluated input
    belongs to emitted context and is committed. A consumed EOS is valid history:
    the target can include that final token; otherwise it ends just before it.
    """

    def decode_causal(
        self,
        target: ModelRuntime[Any, Any],
        sequence: ModelSequence[Any, Any],
        *,
        anchor: int,
        position: int,
        sampler: SequenceSampler,
        allowance: int,
        stop_tokens: tuple[int, ...],
    ) -> Task[CausalResult]: ...
