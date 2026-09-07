"""Bound generation methods own their per-sequence proposal and observation state."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.generation.proposals import Proposal
from magnitude_engine.generation.sampling import SequenceSampler
from magnitude_engine.models.execution import PendingExecution
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.operations import Complete, Task, forward, observe, submit
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

    def __init__(self):
        self._prediction: tuple[mx.array, PendingExecution] | None = None

    def prefill(self, tokens: tuple[int, ...], features: Mapping[str, mx.array]) -> Task[None]:
        yield from ()

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
        remaining: int,
        stop_tokens: tuple[int, ...],
    ) -> Task[CausalResult]:
        if allowance < 1 or sampler.policy.uses_history:
            raise ValueError(
                "causal feedback requires an allowance and history-independent sampling"
            )
        carry = remaining > allowance
        advances = allowance + int(carry) - int(self._prediction is not None)
        if advances:
            target.reserve(sequence, advances)

        def predict(token: mx.array, offset: int) -> Task[tuple[mx.array, PendingExecution]]:
            advance = yield from forward(
                sequence, ModelInputs(token.reshape(1, 1)), ForwardRequest(committed_inputs=1)
            )
            logits = advance.output.logits
            if logits is None or logits.ndim != 3 or logits.shape[:2] != (1, 1):
                raise RuntimeError("causal decode requires one logit vector per input")
            sample = sampler.sample(logits[0, 0], position + offset)
            yield from submit(advance, sample)
            advance.accept_all_lazily()
            return sample, advance.execution

        emitted: list[int] = []
        evaluated = 0
        prediction, self._prediction = self._prediction, None
        if prediction is None:
            prediction = yield from predict(mx.array(anchor, dtype=mx.int32), 0)
            evaluated += 1
        for index in range(allowance):
            token, execution = prediction
            following = None
            if index + 1 < allowance or carry:
                following = yield from predict(token, index + 1)
                evaluated += 1
            # Retire this execution, including all state outputs, without waiting
            # for the successor just submitted. Its leases remain independently held.
            yield Complete(execution)
            yield from observe(token)
            value = cast(int, token.item())
            emitted.append(value)
            if value in stop_tokens:
                if following is not None:
                    yield Complete(following[1])
                break
            if following is not None:
                prediction = following
            if index + 1 == allowance and carry:
                self._prediction = following
        sequence.prune_completed()
        return CausalResult(tuple(emitted), evaluated)

    def checkpoint(self) -> PlainCheckpoint:
        return PlainCheckpoint()

    def close(self) -> None:
        if self._prediction is not None:
            self._prediction[1].complete()
            self._prediction = None


@component("GENERATION:PLAIN:MAG:TARGET")
class PlainMethod:
    identity = "plain"

    def create(
        self, checkpoint: MethodCheckpoint | None = None, *, target: ModelRuntime[Any, Any]
    ) -> PlainSession:
        if checkpoint is not None and not isinstance(checkpoint, PlainCheckpoint):
            raise ValueError("checkpoint belongs to a different generation method")
        return PlainSession()
