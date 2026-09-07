"""Bound generation methods own their per-sequence proposal and observation state."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx

from magnitude_engine import components as c
from magnitude_engine.components import component
from magnitude_engine.generation.proposals import Proposal
from magnitude_engine.generation.sampling import SequenceSampler
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.operations import Task, forward, observe, submit
from magnitude_engine.models.runtime import (
    ForwardRequest,
    ModelRuntime,
    ModelSequence,
)
from magnitude_engine.resources.retention import RetainedStorage

from ..contracts import (
    CausalResult,
    MethodCheckpoint,
    Verification,
)


@dataclass(frozen=True)
class PlainCheckpoint:
    @property
    def reclaimable(self) -> bool:
        return True

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        return ()

    def close(self) -> None:
        pass


class PlainSession:
    features: frozenset[str] = frozenset()
    prefill_features: frozenset[str] = frozenset()

    def prefill(self, tokens: tuple[int, ...], features: Mapping[str, mx.array]) -> None:
        pass

    def propose(self, context: Sequence[int], limit: int) -> Task[Proposal]:
        yield from ()
        return Proposal.from_tokens(())

    def observe(self, verification: Verification) -> None:
        pass

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
    ) -> Task[CausalResult]:
        if allowance < 1 or sampler.policy.uses_history:
            raise ValueError(
                "causal feedback requires an allowance and history-independent sampling"
            )
        target.reserve(sequence, allowance)

        def predict(token: mx.array, offset: int) -> Task[mx.array]:
            advance = yield from forward(
                sequence, ModelInputs(token.reshape(1, 1)), ForwardRequest(committed_inputs=1)
            )
            logits = advance.output.logits
            if logits is None or logits.ndim != 3 or logits.shape[:2] != (1, 1):
                raise RuntimeError("causal decode requires one logit vector per input")
            sample = sampler.sample(logits[0, 0], position + offset)
            yield from submit(advance, sample)
            advance.accept_all_lazily()
            return sample

        emitted: list[int] = []
        evaluated = 1
        token = yield from predict(mx.array(anchor, dtype=mx.int32), 0)
        for index in range(allowance):
            following = None
            if index + 1 < allowance:
                following = yield from predict(token, index + 1)
                evaluated += 1
            yield from observe(token)
            value = cast(int, token.item())
            emitted.append(value)
            if value in stop_tokens:
                break
            if following is not None:
                token = following
        return CausalResult(tuple(emitted), evaluated)

    def checkpoint(self) -> PlainCheckpoint:
        return PlainCheckpoint()

    def close(self) -> None:
        pass


@component(c.GENERATION, source=c.Source.MAG, variant="TARGET")
class PlainMethod:
    identity = "plain"

    def create(
        self, checkpoint: MethodCheckpoint | None = None, *, target: ModelRuntime[Any, Any]
    ) -> PlainSession:
        if checkpoint is not None and not isinstance(checkpoint, PlainCheckpoint):
            raise ValueError("checkpoint belongs to a different generation method")
        return PlainSession()
