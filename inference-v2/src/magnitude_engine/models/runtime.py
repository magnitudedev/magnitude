"""Model execution and transactional state, independent of request scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol, runtime_checkable

import mlx.core as mx

from magnitude_engine.components import component

from .context import InputFactory, InputSource, InputState, ModelCheckpoint, StateCheckpoint
from .execution import (
    ExecutionOwner,
    ExecutionScope,
    PendingExecution,
    ResourceLease,
)
from .inputs import ModelInputs


@dataclass(frozen=True)
class ForwardRequest:
    logits: bool = True
    features: frozenset[str] = frozenset()
    # Inputs in this prefix are known to be causal before execution. A caller may
    # reject only the remaining suffix; publication still waits for reconciliation.
    committed_inputs: int = 0

    def __post_init__(self) -> None:
        if type(self.committed_inputs) is not int or self.committed_inputs < 0:
            raise ValueError("committed input count must be a nonnegative integer")


@dataclass(frozen=True)
class ModelOutput:
    logits: mx.array | None = None
    features: Mapping[str, mx.array] = field(default_factory=dict)

    def arrays(self) -> tuple[mx.array, ...]:
        return (() if self.logits is None else (self.logits,)) + tuple(self.features.values())


class BatchForward[S](Protocol):
    def __call__(
        self,
        inputs: tuple[ModelInputs, ...],
        states: tuple[S, ...],
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        """Execute one physical batch; output arrays have one leading row per state."""
        ...


class ModelProgram[S](Protocol):
    features: frozenset[str]
    conditioning: frozenset[str]

    @property
    def forward_batch(self) -> BatchForward[S] | None: ...

    def forward(
        self, inputs: ModelInputs, state: S, request: ForwardRequest, scope: ExecutionScope
    ) -> ModelOutput: ...


@runtime_checkable
class AllocatedProgram(Protocol):
    """A live continuation keeps its executable allocations available between steps."""

    def acquire(self) -> ResourceLease: ...


class StateTransaction(Protocol):
    def reconcile(self, accepted: int) -> ModelInputs | None:
        """Restore/trim state and return any accepted input that needs replay."""
        ...

    def finish(self, accepted: int) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class LazyCommit(Protocol):
    def commit_all(self) -> None: ...


@runtime_checkable
class RewindableState[S](Protocol):
    def rewind(self, state: S, position: int) -> None: ...


class ModelStateStore[S, C](Protocol):
    def reserve(self, state: S, input_capacity: int) -> None:
        """Prepare address capacity for chained work without advancing committed state."""
        ...

    def create(self, checkpoint: C | None = None) -> S: ...
    def begin(
        self, state: S, inputs: ModelInputs, *, committed_inputs: int = 0
    ) -> StateTransaction: ...
    def arrays(self, state: S) -> tuple[mx.array, ...]: ...
    def checkpoint(self, state: S) -> C: ...
    def release(self, state: S) -> None: ...


@runtime_checkable
class BatchedStateStore[S](Protocol):
    def can_batch(self, states: tuple[S, ...]) -> bool: ...

    def repair_group(self, state: S) -> object:
        """Storage that unresolved advances can share without moving active state."""
        ...

    def prepare_batch(self, states: tuple[S, ...], width: int) -> None:
        """Prepare persistent physical storage before creating row transactions."""
        ...


class ModelSequence[S, C: StateCheckpoint]:
    """A model's sequence state; generation owns its relationship to other models."""

    def __init__(
        self,
        runtime: ModelRuntime[S, C],
        state: S,
        program_lease: ResourceLease | None = None,
        *,
        position: int = 0,
        inputs: InputState | None = None,
    ):
        self.runtime = runtime
        self.state = state
        self.pending: ModelAdvance[S, C] | None = None
        self._committed: list[PendingExecution] = []
        self.closed = False
        self.failed = False
        self._program_lease = program_lease
        self.position = position
        self.inputs = inputs

    def check(self) -> None:
        self.runtime.owner.check()
        if self.closed or self.failed:
            raise RuntimeError("model sequence is unavailable")

    def checkpoint(self) -> ModelCheckpoint[C]:
        self.check()
        if self.pending is not None:
            raise RuntimeError("checkpoint requires reconciled model state")
        self.complete_committed()
        if self.inputs is not None and not self.inputs.boundary(self.position):
            raise ValueError("checkpoint requires an independent input boundary")
        storage = self.runtime.states.checkpoint(self.state)
        try:
            if storage.length != self.position:
                raise RuntimeError("model and storage continuation positions disagree")
            inputs = None if self.inputs is None else self.inputs.checkpoint(self.position)
            return ModelCheckpoint(storage, inputs, self.runtime.checkpoint_domain)
        except BaseException:
            storage.close()
            raise

    def prune_completed(self) -> None:
        self._committed[:] = [work for work in self._committed if not work.done]

    def complete_committed(self) -> bool:
        """Retire committed work; report whether any execution was still pending."""
        self.runtime.owner.check()
        retired = False
        for execution in self._committed:
            retired |= not execution.done
            execution.complete()
        self._committed.clear()
        return retired

    def close(self) -> None:
        if self.closed:
            return
        self.runtime.owner.check()
        # Releasing a row can free addresses in an arena shared with peer work.
        # Complete those consumers before physical storage becomes reusable.
        self.runtime.owner.complete()
        self.complete_committed()
        if self.pending is not None:
            self.pending.complete()
            self.pending.transaction.close()
            self.pending = None
        self.runtime.states.release(self.state)
        if self.inputs is not None:
            self.inputs.close()
        if self._program_lease is not None:
            self._program_lease.close()
            self._program_lease = None
        self.closed = True


class ModelAdvance[S, C: StateCheckpoint]:
    """Lazy output plus an obligation to reconcile exactly one input prefix."""

    def __init__(
        self,
        sequence: ModelSequence[S, C],
        inputs: ModelInputs,
        output: ModelOutput,
        transaction: StateTransaction,
        execution: PendingExecution,
        committed_inputs: int = 0,
    ):
        self.sequence = sequence
        self.inputs = inputs
        self.output = output
        self.transaction = transaction
        self.committed_inputs = committed_inputs
        self.execution = execution
        self.resolved = False
        self._accepted: int | None = None

    def submit(self) -> None:
        self.execution.submit()

    def complete(self) -> None:
        self.execution.complete()
        self.sequence._committed[:] = [
            execution for execution in self.sequence._committed if not execution.done
        ]

    def accept_all_lazily(self) -> None:
        """Commit a supported causal state update while preserving its GPU lifetime.

        This is useful for a chained head: subsequent inputs depend on these arrays,
        and its proposal publication or rewind supplies the completion boundary.
        """
        self.sequence.check()
        if self.resolved or self.sequence.pending is not self:
            raise RuntimeError("model advance is already reconciled")
        if not isinstance(self.transaction, LazyCommit):
            raise ValueError("this state implementation does not support lazy full commit")
        if not self.execution.done:
            self.execution.retain(self.transaction)
        try:
            self.transaction.commit_all()
        except BaseException:
            self.sequence.failed = True
            raise
        if self.execution.done:
            self.transaction.close()
        else:
            self.sequence._committed.append(self.execution)
        self.sequence.pending = None
        self.sequence.position += self.inputs.count
        self.resolved = True

    def accept(self, count: int) -> None:
        replay = self.prepare_accept(count)
        if replay is not None:
            self.sequence.runtime.replay((self,), (replay,))
        self.finish_accept(count)

    def prepare_accept(self, count: int) -> ModelInputs | None:
        """Reconcile row state, exposing neural repair to the operation executor."""
        self.sequence.check()
        if self.resolved or self.sequence.pending is not self:
            raise RuntimeError("model advance is already reconciled")
        if not self.committed_inputs <= count <= self.inputs.count:
            raise ValueError("accepted length is outside the model advance commitment")
        if self.sequence.inputs is not None and not self.sequence.inputs.boundary(
            self.sequence.position + count
        ):
            raise ValueError("accepted length is not an independent input boundary")
        try:
            self.complete()
            try:
                replay = self.transaction.reconcile(count)
            except BaseException as error:
                self.sequence.runtime.owner.drain_after_failure(error)
                raise
            self._accepted = count
            return replay
        except BaseException:
            self.sequence.failed = True
            raise

    def finish_accept(self, count: int) -> None:
        self.sequence.check()
        if self.resolved or self.sequence.pending is not self or count != self._accepted:
            raise ValueError("commit must finish the prepared accepted prefix exactly once")
        try:
            self.transaction.finish(count)
            self.transaction.close()
        except BaseException:
            self.sequence.failed = True
            raise
        self.resolved = True
        self.sequence.pending = None
        self.sequence.position += count


@component("MODEL:EXECUTOR:MAG:STANDARD")
class ModelRuntime[S, C: StateCheckpoint]:
    def __init__(
        self,
        program: ModelProgram[S],
        states: ModelStateStore[S, C],
        owner: ExecutionOwner,
        *,
        preparation: InputFactory | None = None,
    ):
        self.program = program
        self.states = states
        self.owner = owner
        self.preparation = preparation
        # Live state handles cannot cross a model residency. Persisted state needs
        # a separately validated import operation, not reuse of an in-memory handle.
        self.checkpoint_domain = object()

    def replay(
        self, advances: tuple[ModelAdvance[S, C], ...], inputs: tuple[ModelInputs, ...]
    ) -> None:
        """Repair accepted prefixes inside their original state transactions."""
        self.owner.check()
        if not advances or len(advances) != len(inputs):
            raise ValueError("repair requires one accepted prefix per advance")
        if len({id(a.sequence) for a in advances}) != len(advances):
            raise ValueError("repair requires distinct sequences")
        for advance, row in zip(advances, inputs, strict=True):
            advance.sequence.check()
            if (
                advance.sequence.runtime is not self
                or advance.resolved
                or advance.sequence.pending is not advance
                or not 0 < row.count == advance._accepted
                or row.count != inputs[0].count
            ):
                raise ValueError("repair requires compatible unresolved advances")
        batch = self.program.forward_batch
        groups = tuple(self.repair_group(a.sequence) for a in advances)
        if len(advances) > 1 and (batch is None or any(g is not groups[0] for g in groups)):
            for advance, row in zip(advances, inputs, strict=True):
                self.replay((advance,), (row,))
            return
        states = tuple(a.sequence.state for a in advances)
        try:
            with self.owner.scope() as scope:
                for advance, row in zip(advances, inputs, strict=True):
                    sequence = advance.sequence
                    if sequence.inputs is not None:
                        scope.acquire(
                            partial(sequence.inputs.acquire, sequence.position, row.count)
                        )
                if len(advances) == 1:
                    self.program.forward(inputs[0], states[0], ForwardRequest(False), scope)
                else:
                    assert batch is not None
                    batch(inputs, states, ForwardRequest(False), scope)
                scope.seal(*(a for state in states for a in self.states.arrays(state))).complete()
        except BaseException:
            for advance in advances:
                advance.sequence.failed = True
            raise

    def create(
        self, checkpoint: ModelCheckpoint[C] | None = None, *, inputs: InputSource | None = None
    ) -> ModelSequence[S, C]:
        self.owner.check()
        if checkpoint is not None and (
            checkpoint.closed or checkpoint.domain is not self.checkpoint_domain
        ):
            raise ValueError("checkpoint belongs to another model residency or is closed")
        lease = self.program.acquire() if isinstance(self.program, AllocatedProgram) else None
        context = None
        try:
            previous = None if checkpoint is None else checkpoint.inputs
            context = (
                inputs.bind(previous)
                if inputs is not None
                else None
                if previous is None
                else previous.restore()
            )
            return ModelSequence(
                self,
                self.states.create(None if checkpoint is None else checkpoint.storage),
                lease,
                position=0 if checkpoint is None else checkpoint.length,
                inputs=context,
            )
        except BaseException:
            if context is not None:
                context.close()
            if lease is not None:
                lease.close()
            raise

    def reserve(self, sequence: ModelSequence[S, C], input_capacity: int) -> None:
        sequence.check()
        if sequence.runtime is not self or sequence.pending is not None:
            raise ValueError("capacity preparation requires this model's idle sequence")
        if type(input_capacity) is not int or input_capacity < 1:
            raise ValueError("model input capacity must be a positive integer")
        try:
            self.states.reserve(sequence.state, input_capacity)
        except MemoryError:
            # A physical allocation may require retiring pins from any peer in
            # the shared arena. No neural work or sampling has started here.
            if not self.owner.complete():
                raise
            sequence.prune_completed()
            self.states.reserve(sequence.state, input_capacity)

    def can_batch(self, sequences: tuple[ModelSequence[S, C], ...]) -> bool:
        return self.program.forward_batch is not None and (
            not isinstance(self.states, BatchedStateStore)
            or self.states.can_batch(tuple(s.state for s in sequences))
        )

    def repair_group(self, sequence: ModelSequence[S, C]) -> object:
        return (
            self.states.repair_group(sequence.state)
            if isinstance(self.states, BatchedStateStore)
            else self
        )

    def input_key(self, sequence: ModelSequence[S, C], count: int) -> object:
        return (
            None if sequence.inputs is None else sequence.inputs.batch_key(sequence.position, count)
        )

    def _validate_boundaries(
        self, sequence: ModelSequence[S, C], count: int, committed: int
    ) -> None:
        if sequence.inputs is not None and not all(
            sequence.inputs.boundary(sequence.position + offset) for offset in (0, committed, count)
        ):
            raise ValueError("model advancement requires independent input boundaries")

    def rewind(self, sequence: ModelSequence[S, C], position: int) -> None:
        sequence.check()
        if sequence.runtime is not self or sequence.pending is not None:
            raise ValueError("rewind requires this model's idle sequence")
        if not isinstance(self.states, RewindableState):
            raise ValueError("this model state does not support direct rewind")
        if type(position) is not int or not 0 <= position <= sequence.position:
            raise ValueError("rewind position must be inside committed model history")
        self._validate_boundaries(sequence, 0, 0)
        if sequence.inputs is not None and not sequence.inputs.boundary(position):
            raise ValueError("rewind requires an independent input boundary")
        sequence.complete_committed()
        self.states.rewind(sequence.state, position)
        sequence.position = position

    def forward(
        self,
        sequence: ModelSequence[S, C],
        inputs: ModelInputs | tuple[int, ...],
        request: ForwardRequest | None = None,
    ) -> ModelAdvance[S, C]:
        sequence.check()
        request = request or ForwardRequest()
        if sequence.runtime is not self or sequence.pending is not None:
            raise ValueError("forward needs an idle sequence owned by this runtime")
        if isinstance(inputs, tuple):
            inputs = ModelInputs.from_tokens(inputs)
        if not inputs.count or request.committed_inputs > inputs.count:
            raise ValueError("model input must be nonempty and contain its committed prefix")
        if set(inputs.conditioning) != self.program.conditioning:
            raise ValueError("model conditioning differs from its declared inputs")
        if not request.features <= self.program.features:
            raise ValueError("model program does not provide the requested features")
        self._validate_boundaries(sequence, inputs.count, request.committed_inputs)
        if isinstance(self.states, BatchedStateStore):
            self.states.prepare_batch((sequence.state,), inputs.count)
        if sequence.inputs is not None:
            inputs = sequence.inputs.assemble(inputs, sequence.position)
        transaction = self.states.begin(
            sequence.state, inputs, committed_inputs=request.committed_inputs
        )
        try:
            with self.owner.scope() as scope:
                if sequence.inputs is not None:
                    scope.acquire(partial(sequence.inputs.acquire, sequence.position, inputs.count))
                output = self.program.forward(inputs, sequence.state, request, scope)
                if request.logits and output.logits is None:
                    raise RuntimeError("model program omitted requested logits")
                if not request.features <= output.features.keys():
                    raise RuntimeError("model program omitted requested features")
                execution = scope.seal(*output.arrays(), *self.states.arrays(sequence.state))
        except BaseException:
            sequence.failed = True
            if not self.owner.requires_disposal:
                transaction.close()
            raise
        advance = ModelAdvance(
            sequence, inputs, output, transaction, execution, request.committed_inputs
        )
        sequence.pending = advance
        return advance

    def forward_batch(
        self,
        sequences: tuple[ModelSequence[S, C], ...],
        inputs: tuple[ModelInputs, ...],
        request: ForwardRequest | None = None,
    ) -> tuple[ModelAdvance[S, C], ...]:
        """Group neural work while keeping reconciliation and state ownership per row.

        All rows reserve state before any program execution. They share a GPU
        completion obligation, but each row resolves independently. State storage
        owns and accounts for any shared backing through its last logical or
        execution lease. Completing/cancelling one row does not reconcile peers.
        Equal input widths are required; context positions may differ.
        """
        self.owner.check()
        request = request or ForwardRequest()
        forward = self.program.forward_batch
        if forward is None:
            raise ValueError("this model program does not support physical batching")
        if not sequences or len(sequences) != len(inputs):
            raise ValueError("model batch requires one input per sequence")
        if len({id(sequence) for sequence in sequences}) != len(sequences):
            raise ValueError("model batch requires distinct sequences")
        for sequence, row in zip(sequences, inputs, strict=True):
            sequence.check()
            if sequence.runtime is not self or sequence.pending is not None:
                raise ValueError("batch forward needs idle sequences owned by this runtime")
            if (
                not row.count
                or row.count != inputs[0].count
                or request.committed_inputs > row.count
            ):
                raise ValueError("model batch inputs require equal nonempty widths")
            if set(row.conditioning) != self.program.conditioning:
                raise ValueError("model conditioning differs from its declared inputs")
            self._validate_boundaries(sequence, row.count, request.committed_inputs)
        key = self.input_key(sequences[0], inputs[0].count)
        if any(
            self.input_key(s, row.count) != key for s, row in zip(sequences, inputs, strict=True)
        ):
            raise ValueError("model batch requires compatible input semantics")
        if not request.features <= self.program.features:
            raise ValueError("model program does not provide the requested features")
        if isinstance(self.states, BatchedStateStore):
            self.states.prepare_batch(tuple(s.state for s in sequences), inputs[0].count)
        transactions: list[StateTransaction] = []
        inputs = tuple(
            row if sequence.inputs is None else sequence.inputs.assemble(row, sequence.position)
            for sequence, row in zip(sequences, inputs, strict=True)
        )
        try:
            for sequence, row in zip(sequences, inputs, strict=True):
                transactions.append(
                    self.states.begin(
                        sequence.state, row, committed_inputs=request.committed_inputs
                    )
                )
        except BaseException:
            for transaction in reversed(transactions):
                transaction.close()
            raise
        try:
            with self.owner.scope() as scope:
                for sequence, row in zip(sequences, inputs, strict=True):
                    if sequence.inputs is not None:
                        scope.acquire(
                            partial(sequence.inputs.acquire, sequence.position, row.count)
                        )
                output = forward(inputs, tuple(s.state for s in sequences), request, scope)
                if request.logits and output.logits is None:
                    raise RuntimeError("batched model omitted requested logits")
                if not request.features <= output.features.keys():
                    raise RuntimeError("batched model omitted requested features")
                if any(
                    a.ndim < 2 or a.shape[:2] != (len(sequences), inputs[0].count)
                    for a in output.arrays()
                ):
                    raise RuntimeError("batched model output does not align with its inputs")
                rows = tuple(
                    ModelOutput(
                        None if output.logits is None else mx.array(output.logits[i : i + 1]),
                        {
                            name: mx.array(value[i : i + 1])
                            for name, value in output.features.items()
                        },
                    )
                    for i in range(len(sequences))
                )
                execution = scope.seal(
                    *(a for row in rows for a in row.arrays()),
                    *(a for sequence in sequences for a in self.states.arrays(sequence.state)),
                )
        except BaseException:
            for sequence, transaction in zip(sequences, transactions, strict=True):
                sequence.failed = True
                if not self.owner.requires_disposal:
                    transaction.close()
            raise
        advances = tuple(
            ModelAdvance(sequence, row, output, transaction, execution, request.committed_inputs)
            for sequence, row, output, transaction in zip(
                sequences, inputs, rows, transactions, strict=True
            )
        )
        for sequence, advance in zip(sequences, advances, strict=True):
            sequence.pending = advance
        return advances

    def prefill(
        self,
        sequence: ModelSequence[S, C],
        inputs: ModelInputs | tuple[int, ...],
        features: frozenset[str] = frozenset(),
    ) -> Mapping[str, mx.array]:
        if isinstance(inputs, tuple):
            inputs = ModelInputs.from_tokens(inputs)
        if not inputs.count:
            return {}
        advance = self.forward(
            sequence, inputs, ForwardRequest(False, features, committed_inputs=inputs.count)
        )
        advance.accept(inputs.count)
        return advance.output.features
