"""Qwen MTP proposal state, target-feature alignment and deferred observation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.generation.features import RetainedFeature
from magnitude_engine.generation.proposals import Proposal
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.operations import Task, complete, forward, project_vocabulary, submit
from magnitude_engine.models.runtime import ForwardRequest, ModelAdvance, ModelRuntime
from magnitude_engine.models.state.native import LibraryCheckpoint, LibraryState
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.retention import RetainedStorage

from ..contracts import MethodCheckpoint, Verification

HeadRuntime = ModelRuntime[LibraryState, LibraryCheckpoint]


class MTPCheckpoint:
    @property
    def reclaimable(self) -> bool:
        return self.head is None or self.head.reclaimable

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        return (
            *(() if self.head is None else self.head.retained_storage()),
            RetainedStorage(self, self.reservation.size),
        )

    def __init__(self, session: MTPSession):
        self.owner = session.binding
        self.closed = False
        tensors = ([session.pending.value] if session.pending is not None else []) + [
            h.value for _, h in session.buffer
        ]
        self.reservation = session.binding.budget.reserve(
            "mtp-checkpoint", sum(a.nbytes for a in tensors)
        )
        self.head: LibraryCheckpoint | None = None
        self.pending: mx.array | None = None
        self.buffer: list[tuple[int, mx.array]] = []
        self.position = session.position
        try:
            self.head = session.row.checkpoint()
            self.pending = None if session.pending is None else mx.array(session.pending.value)
            self.buffer = [(token, mx.array(hidden.value)) for token, hidden in session.buffer]
            mx.eval(*([] if self.pending is None else [self.pending]), *(h for _, h in self.buffer))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        if self.head is not None:
            self.head.close()
        self.pending = None
        self.buffer.clear()
        self.reservation.close()
        self.closed = True


class MTPSession:
    """Head history follows committed inputs; the final target feature awaits its successor.

    A checkpoint contains no token beyond its target prefix. Known committed pairs may
    wait in buffer, but the next anchor is supplied by the actual continuation.
    """

    def __init__(self, binding: MTPMethod, checkpoint: MTPCheckpoint | None):
        self.binding = binding
        self.features = self.prefill_features = frozenset({binding.target_feature})
        self.row = binding.head.create(None if checkpoint is None else checkpoint.head)
        self.position = 0 if checkpoint is None else checkpoint.position
        self.pending: RetainedFeature | None = None
        self.buffer: list[tuple[int, RetainedFeature]] = []
        self.appended = 0
        self.proposed: Proposal | None = None
        self.closed = False
        try:
            if checkpoint is not None:
                if checkpoint.pending is not None:
                    self.pending = RetainedFeature(checkpoint.pending, binding.budget)
                for token, value in checkpoint.buffer:
                    self.buffer.append((token, RetainedFeature(value, binding.budget)))
        except BaseException:
            self.close()
            raise

    def _features(self, tokens: tuple[int, ...], features: Mapping[str, mx.array]) -> mx.array:
        hidden = features[self.binding.target_feature]
        if not tokens or hidden.shape[:2] != (1, len(tokens)):
            raise ValueError("target features do not align with consumed inputs")
        return hidden

    def prefill(self, tokens: tuple[int, ...], features: Mapping[str, mx.array]) -> Task[None]:
        if self.closed or self.proposed is not None:
            raise RuntimeError("MTP prefill requires an idle live method state")
        hidden = self._features(tokens, features)
        yield from self._flush()
        previous = hidden[:, :-1]
        shifted = tokens[1:]
        consumed = ()
        if self.pending is not None:
            previous = mx.concatenate([self.pending.value, previous], axis=1)
            shifted = tokens
            consumed = (self.pending,)
        replacement = RetainedFeature(hidden[:, -1:], self.binding.budget)
        try:
            if shifted:
                advance = yield from self._forward(
                    mx.array([shifted], dtype=mx.int32), previous, consumed
                )
                yield from complete(advance)
            self.pending = replacement
            self.row.complete_committed()
        except BaseException:
            replacement.close()
            raise

    def _forward(
        self, tokens: mx.array, previous: mx.array, consumed: tuple[RetainedFeature, ...] = ()
    ) -> Task[ModelAdvance[LibraryState, LibraryCheckpoint]]:
        advance = yield from forward(
            self.row,
            ModelInputs(tokens, {"previous_hidden": previous}),
            ForwardRequest(False, frozenset({"draft_hidden"}), committed_inputs=tokens.shape[1]),
        )
        for feature in consumed:
            advance.execution.retain(feature)
        yield from submit(advance)
        advance.accept_all_lazily()
        self.position += tokens.shape[1]
        return advance

    def _flush(self) -> Task[None]:
        if not self.buffer:
            return
        tokens = mx.array([[token for token, _ in self.buffer]], dtype=mx.int32)
        previous = mx.concatenate([hidden.value for _, hidden in self.buffer], axis=1)
        yield from self._forward(tokens, previous, tuple(value for _, value in self.buffer))
        self.buffer.clear()

    def propose(self, context: Sequence[int], limit: int) -> Task[Proposal]:
        if self.closed or self.proposed is not None:
            raise RuntimeError("MTP proposal requires an idle live method state")
        if limit <= 0 or self.pending is None:
            return Proposal.from_tokens(())
        yield from self._flush()
        advance = yield from self._forward(
            mx.array([[context[-1]]], dtype=mx.int32), self.pending.value, (self.pending,)
        )
        self.pending = None
        hidden = advance.output.features["draft_hidden"]
        tokens = []
        for index in range(min(limit, self.binding.capacity)):
            if index:
                advance = yield from self._forward(tokens[-1], hidden)
                hidden = advance.output.features["draft_hidden"]
                self.appended += 1
            logits = yield from project_vocabulary(self.binding.project, hidden[:, -1:])
            tokens.append(mx.argmax(logits, axis=-1).astype(mx.int32))
        output = mx.concatenate(tokens, axis=1)
        self.proposed = Proposal(output.reshape(-1))
        return self.proposed

    def observe(self, verification: Verification) -> None:
        if self.closed:
            raise RuntimeError("MTP state is closed")
        if self.proposed is not None and verification.inputs[1:] != self.proposed.host():
            raise ValueError("MTP observation differs from its proposal")
        if self.proposed is None and verification.accepted_inputs != len(verification.inputs):
            raise ValueError("externally determined advancement must be fully committed")
        accepted = verification.accepted_inputs - 1
        if not 0 <= accepted < len(verification.inputs):
            raise ValueError("MTP observation has an invalid accepted prefix")
        hidden = self._features(verification.inputs, verification.features)
        self.row.complete_committed()
        keep = min(accepted, self.appended)
        drop = self.appended - keep
        if drop:
            self.binding.head.rewind(self.row, self.position - drop)
            self.position -= drop
        # Without drafting (one output slot or forced inputs), the anchor still
        # needs its preceding target feature. Never store the unpublished successor.
        if self.pending is not None:
            self.buffer.append((verification.inputs[0], self.pending))
            self.pending = None
        self.appended = 0
        self.proposed = None
        for i in range(keep, accepted):
            self.buffer.append(
                (
                    verification.inputs[i + 1],
                    RetainedFeature(hidden[:, i : i + 1], self.binding.budget),
                )
            )
        self.pending = RetainedFeature(hidden[:, accepted : accepted + 1], self.binding.budget)

    def checkpoint(self) -> MTPCheckpoint:
        if self.closed or self.proposed is not None:
            raise RuntimeError("MTP checkpoint requires reconciled proposal state")
        return MTPCheckpoint(self)

    def close(self) -> None:
        if self.closed:
            return
        self.row.close()
        if self.pending is not None:
            self.pending.close()
            self.pending = None
        for _, feature in self.buffer:
            feature.close()
        self.buffer.clear()
        self.closed = True


@component("GENERATION:SPECULATION:MAG:TARGET_MATCHING")
class MTPMethod:
    def __init__(
        self,
        *,
        target: ModelRuntime[Any, Any],
        head: HeadRuntime,
        target_feature: str,
        project: Callable[[mx.array], mx.array],
        capacity: int,
        budget: MemoryBudget,
        identity: str,
    ):
        if capacity < 1 or target_feature not in target.program.features or not identity:
            raise ValueError("invalid MTP capacity, target feature or artifact identity")
        if head.owner is not target.owner:
            raise ValueError("target and MTP head must share an execution owner")
        if (
            head.program.conditioning != frozenset({"previous_hidden"})
            or "draft_hidden" not in head.program.features
        ):
            raise ValueError(
                "MTP head does not provide the required conditioned execution contract"
            )
        self.target, self.head = target, head
        self.target_feature, self.project = target_feature, project
        self.capacity, self.budget = capacity, budget
        self.artifact_path = identity
        self.identity = f"mtp:{identity}:{capacity}:{target_feature}"

    def create(
        self, checkpoint: MethodCheckpoint | None = None, *, target: ModelRuntime[Any, Any]
    ) -> MTPSession:
        if target is not self.target:
            raise ValueError("MTP binding belongs to another target runtime")
        if checkpoint is not None and (
            not isinstance(checkpoint, MTPCheckpoint)
            or checkpoint.closed
            or checkpoint.owner is not self
        ):
            raise ValueError("MTP checkpoint belongs to another binding or is closed")
        return MTPSession(self, checkpoint)
