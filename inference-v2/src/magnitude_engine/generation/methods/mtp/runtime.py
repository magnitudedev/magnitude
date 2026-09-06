"""Qwen MTP proposal state, target-feature alignment and deferred observation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx

from magnitude_engine.generation.features import RetainedFeature
from magnitude_engine.generation.proposals import Proposal
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.operations import Task, forward, project_vocabulary, submit
from magnitude_engine.models.runtime import ForwardRequest, ModelRuntime
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
        tensors = ([session.seed.value] if session.seed is not None else []) + [
            h.value for _, h in session.buffer
        ]
        self.reservation = session.binding.budget.reserve(
            "mtp-checkpoint", sum(a.nbytes for a in tensors)
        )
        self.head: LibraryCheckpoint | None = None
        self.seed: mx.array | None = None
        self.buffer: list[tuple[int, mx.array]] = []
        self.position = session.position
        try:
            self.head = session.row.checkpoint()
            self.seed = None if session.seed is None else mx.array(session.seed.value)
            self.buffer = [(token, mx.array(hidden.value)) for token, hidden in session.buffer]
            mx.eval(*([] if self.seed is None else [self.seed]), *(h for _, h in self.buffer))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        if self.head is not None:
            self.head.close()
        self.seed = None
        self.buffer.clear()
        self.reservation.close()
        self.closed = True


class MTPSession:
    prefill_features: frozenset[str] = frozenset()

    def __init__(self, binding: MTPMethod, checkpoint: MTPCheckpoint | None):
        self.binding = binding
        self.features = frozenset({binding.target_feature})
        self.row = binding.head.create(None if checkpoint is None else checkpoint.head)
        self.position = 0 if checkpoint is None else checkpoint.position
        self.seed: RetainedFeature | None = None
        self.buffer: list[tuple[int, RetainedFeature]] = []
        self.appended = 0
        self.proposed: Proposal | None = None
        self.proposal: RetainedFeature | None = None
        self.closed = False
        try:
            if checkpoint is not None:
                if checkpoint.seed is not None:
                    self.seed = RetainedFeature(checkpoint.seed, binding.budget)
                for token, value in checkpoint.buffer:
                    self.buffer.append((token, RetainedFeature(value, binding.budget)))
        except BaseException:
            self.close()
            raise

    def prefill(self, tokens: tuple[int, ...], features: Mapping[str, mx.array]) -> None:
        # Released Qwen MTP starts from the first target decode, not prompt-head prefill.
        pass

    def _forward(
        self, tokens: mx.array, previous: mx.array, consumed: tuple[RetainedFeature, ...] = ()
    ) -> Task[mx.array]:
        advance = yield from forward(
            self.row,
            ModelInputs(tokens, {"previous_hidden": previous}),
            ForwardRequest(False, frozenset({"draft_hidden"})),
        )
        for feature in consumed:
            advance.execution.retain(feature)
        yield from submit(advance)
        advance.accept_all_lazily()
        self.position += tokens.shape[1]
        return advance.output.features["draft_hidden"]

    def _flush(self) -> Task[None]:
        if not self.buffer:
            return
        tokens = mx.array([[token for token, _ in self.buffer]], dtype=mx.int32)
        previous = mx.concatenate([hidden.value for _, hidden in self.buffer], axis=1)
        hidden = yield from self._forward(
            tokens, previous, tuple(value for _, value in self.buffer)
        )
        self.buffer.clear()
        replacement = RetainedFeature(hidden[:, -1:], self.binding.budget)
        if self.seed is not None:
            self.seed.close()
        self.seed = replacement

    def propose(self, context: Sequence[int], limit: int) -> Task[Proposal]:
        if self.closed or self.proposed is not None:
            raise RuntimeError("MTP proposal requires an idle live method state")
        if limit <= 0:
            return Proposal.from_tokens(())
        yield from self._flush()
        if self.seed is None:
            return Proposal.from_tokens(())
        width = min(limit, self.binding.capacity)
        hidden = self.seed.value
        tokens, learned = [], []
        for index in range(width):
            if index:
                hidden = yield from self._forward(tokens[-1], hidden)
                self.appended += 1
            learned.append(hidden[:, -1:])
            logits = yield from project_vocabulary(self.binding.project, hidden[:, -1:])
            tokens.append(mx.argmax(logits, axis=-1).astype(mx.int32))
        replacement = RetainedFeature(mx.concatenate(learned, axis=1), self.binding.budget)
        if self.proposal is not None:
            self.proposal.close()
        self.proposal = replacement
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
        hidden = verification.features[self.binding.target_feature]
        if hidden.shape[:2] != (1, len(verification.inputs)):
            raise ValueError("target features do not align with verification inputs")
        self.row.complete_committed()
        keep = min(accepted, self.appended)
        drop = self.appended - keep
        if drop:
            self.binding.head.rewind(self.row, self.position - drop)
            self.position -= drop
        self.appended = 0
        self.proposed = None
        for i in range(keep, accepted):
            self.buffer.append(
                (
                    verification.inputs[i + 1],
                    RetainedFeature(hidden[:, i : i + 1], self.binding.budget),
                )
            )
        self.buffer.append(
            (
                verification.next_token,
                RetainedFeature(hidden[:, accepted : accepted + 1], self.binding.budget),
            )
        )

    def checkpoint(self) -> MTPCheckpoint:
        if self.closed or self.proposed is not None:
            raise RuntimeError("MTP checkpoint requires reconciled proposal state")
        return MTPCheckpoint(self)

    def close(self) -> None:
        if self.closed:
            return
        self.row.close()
        if self.seed is not None:
            self.seed.close()
            self.seed = None
        for _, feature in self.buffer:
            feature.close()
        self.buffer.clear()
        if self.proposal is not None:
            self.proposal.close()
            self.proposal = None
        self.closed = True


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
