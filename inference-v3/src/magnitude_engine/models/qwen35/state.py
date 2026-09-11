"""Numerical Qwen continuation state and isolated candidate advances.

Input/conditioning history and publication belong to the enclosing model and
generation continuations. A snapshot here is explicitly numerical state only.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

from magnitude_engine.kernels.precision import NATIVE_BF16, Precision
from magnitude_engine.models.qwen35.description import Geometry, MixerKind
from magnitude_engine.platform.execution import (
    DeviceContext,
    DType,
    Tensor,
    TensorSpec,
    Ticket,
    reclaimable_bytes,
)
from magnitude_engine.state.kv import KVLayout, KVPool, KVSpan, KVWrite, physical_runs


@dataclass(frozen=True)
class RecurrentState:
    convolution: Tensor
    delta: Tensor

    def fork(self) -> RecurrentState:
        convolution = self.convolution.view(self.convolution.spec)
        try:
            return RecurrentState(convolution, self.delta.view(self.delta.spec))
        except BaseException:
            convolution.close()
            raise

    def close(self) -> None:
        self.convolution.close()
        self.delta.close()


@dataclass(frozen=True)
class RecurrentAdvance:
    previous: RecurrentState
    following: RecurrentState


def _fork_data(kv: tuple[KVSpan, ...], recurrent: tuple[RecurrentState | None, ...]):
    with ExitStack() as cleanup:
        spans = []
        for span in kv:
            run = span.run.fork()
            cleanup.callback(run.close)
            spans.append(KVSpan(run, span.start, span.length))
        states = []
        for state in recurrent:
            value = None if state is None else state.fork()
            if value is not None:
                cleanup.callback(value.close)
            states.append(value)
        cleanup.pop_all()
        return tuple(spans), tuple(states)


def _close_data(kv: tuple[KVSpan, ...], recurrent: tuple[RecurrentState | None, ...]) -> None:
    for span in kv:
        span.run.close()
    for state in recurrent:
        if state is not None:
            state.close()


class QwenState:
    def __init__(
        self,
        store: QwenStateStore,
        position: int,
        kv: tuple[KVSpan, ...],
        recurrent: tuple[RecurrentState | None, ...],
    ):
        self.store, self.position = store, position
        self.kv, self.recurrent = kv, recurrent
        self.expected_end = position
        self.pending: QwenAdvance | None = None
        self.closed = False

    def check(self) -> None:
        self.store.context.check()
        if self.closed or self.store.closed:
            raise RuntimeError("Qwen state is closed")

    def anticipate(self, position: int) -> None:
        """Supply a known input horizon without allocating or claiming future KV."""
        self.check()
        if type(position) is not int or not 0 <= position <= self.store.geometry.context_limit:
            raise ValueError("anticipated input exceeds the model context")
        self.expected_end = max(self.expected_end, position)

    def begin(self, count: int) -> QwenAdvance:
        self.check()
        if self.pending is not None:
            raise RuntimeError("state already has an unresolved advance")
        if (
            type(count) is not int
            or count <= 0
            or self.position + count > self.store.geometry.context_limit
        ):
            raise ValueError("input advance exceeds the model context")
        result = QwenAdvance(self, count)
        self.pending = result
        return result

    def checkpoint(self) -> QwenCheckpoint:
        self.check()
        if self.pending is not None:
            raise RuntimeError("cannot checkpoint an unresolved advance")
        kv, recurrent = _fork_data(self.kv, self.recurrent)
        checkpoint = QwenCheckpoint(self.store, self.position, kv, recurrent)
        self.store._checkpoints.add(checkpoint)
        return checkpoint

    def close(self) -> None:
        self.store.context.check_thread()
        if not self.closed:
            if self.pending is not None:
                self.pending.abort()
            _close_data(self.kv, self.recurrent)
            self.closed = True
            self.store._states.discard(self)


class QwenCheckpoint:
    def __init__(
        self,
        store: QwenStateStore,
        position: int,
        kv: tuple[KVSpan, ...],
        recurrent: tuple[RecurrentState | None, ...],
    ):
        self.store, self.position, self.kv, self.recurrent = store, position, kv, recurrent
        self.closed = False

    def close(self) -> None:
        self.store.context.check_thread()
        if not self.closed:
            _close_data(self.kv, self.recurrent)
            self.closed = True
            self.store._checkpoints.discard(self)


class QwenAdvance:
    def __init__(self, state: QwenState, count: int):
        self.state, self.count, self.position = state, count, state.position
        self.closed = False
        private_tail = bool(state.kv) and state.kv[-1].run.exclusive
        self._completion: Ticket | None = None
        with ExitStack() as cleanup:
            existing_banks = frozenset(state.store._banks)
            # Register first so resource consumers unwind before idle banks are
            # reclaimed. Successful submissions may retain banks for reuse.
            cleanup.callback(
                lambda: (
                    state.store._release_new_idle(existing_banks)
                    if self._completion is None
                    else None
                )
            )
            # Reserve mandatory next-state banks before the KV allocator uses
            # remaining capacity for optional slab growth.
            advances = []
            for previous in state.recurrent:
                if previous is None:
                    advances.append(None)
                    continue
                original = previous.fork()
                cleanup.callback(original.close)
                following = state.store._recurrent(zero=False)
                cleanup.callback(following.close)
                advances.append(RecurrentAdvance(original, following))
            spans = []
            for span in state.kv:
                run = span.run.fork()
                cleanup.callback(run.close)
                spans.append(KVSpan(run, span.start, span.length))
            writes = []
            consumed = 0
            if private_tail:
                tail = spans[-1]
                added = min(count, tail.run.capacity - tail.length)
                if added:
                    writes.append(KVWrite(tail.run, tail.length, 0, added))
                    spans[-1] = KVSpan(tail.run, tail.start, tail.length + added)
                    consumed = added
            if consumed < count and state.store.pool is not None:
                runs = state.store.pool.reserve(
                    count - consumed,
                    after=spans[-1].run if spans else None,
                    preferred_tokens=max(
                        0,
                        min(
                            state.store.geometry.context_limit,
                            state.expected_end + state.store.pool.page_tokens,
                        )
                        - self.position
                        - consumed,
                    ),
                )
                for run in runs:
                    cleanup.callback(run.close)
                for run in runs:
                    length = min(count - consumed, run.capacity)
                    spans.append(KVSpan(run, self.position + consumed, length))
                    writes.append(KVWrite(run, 0, consumed, length))
                    consumed += length
            self.kv, self.writes, self.recurrent = tuple(spans), tuple(writes), tuple(advances)
            self.reads = physical_runs(self.kv)
            self._cleanup = cleanup.pop_all()

    def submitted(self, completion: Ticket) -> None:
        if self.closed or self._completion is not None or self.state.pending is not self:
            raise RuntimeError("advance is not awaiting submission")
        if completion.context is not self.state.store.context:
            raise ValueError("advance was submitted on another execution owner")
        self._completion = completion

    def commit(self) -> None:
        self.state.check()
        if self.closed or self.state.pending is not self:
            raise RuntimeError("advance is no longer current")
        completion = self._completion
        if completion is None or not completion.done or self.state.position != self.position:
            raise RuntimeError("state commit requires proven completion on its execution owner")
        # wait() also propagates a terminal submission error on an already done ticket.
        completion.wait()
        recurrent = tuple(None if item is None else item.following for item in self.recurrent)
        kv, retained = _fork_data(self.kv, recurrent)
        _close_data(self.state.kv, self.state.recurrent)
        self.state.kv, self.state.recurrent = kv, retained
        self.state.position += self.count
        self.state.pending = None
        self._cleanup.close()
        self.closed = True

    def abort(self) -> None:
        self.state.store.context.check_thread()
        if not self.closed:
            if self.state.pending is self:
                self.state.pending = None
            self._cleanup.close()
            self.closed = True


class QwenStateStore:
    def __init__(
        self,
        context: DeviceContext,
        geometry: Geometry,
        precision: Precision = NATIVE_BF16,
    ):
        self.context, self.geometry, self.precision = context, geometry, precision
        attention_layers = sum(kind == MixerKind.ATTENTION for kind in geometry.layers)
        self.pool = (
            KVPool(
                context,
                KVLayout(
                    attention_layers,
                    geometry.kv_heads,
                    geometry.attention_width,
                    precision.kv,
                ),
            )
            if attention_layers
            else None
        )
        self._states: set[QwenState] = set()
        self._checkpoints: set[QwenCheckpoint] = set()
        self._banks: list[RecurrentState] = []
        self.closed = False

    def _recurrent(self, *, zero: bool) -> RecurrentState:
        for bank in self._banks:
            if bank.convolution.exclusive_allocation and bank.delta.exclusive_allocation:
                if zero:
                    self.context.initialize_zero(bank.convolution)
                    self.context.initialize_zero(bank.delta)
                return bank.fork()
        g = self.geometry
        allocate = self.context.zeros if zero else self.context.allocate
        with ExitStack() as cleanup:
            convolution = allocate(
                TensorSpec(
                    (1, g.recurrent_channels, g.convolution_width - 1), self.precision.activation
                )
            )
            cleanup.callback(convolution.close)
            delta = allocate(
                TensorSpec(
                    (1, g.recurrent_value_heads, g.recurrent_width, g.recurrent_width), DType.F32
                )
            )
            cleanup.callback(delta.close)
            bank = RecurrentState(convolution, delta)
            loan = bank.fork()
            self._banks.append(bank)
            cleanup.pop_all()
            return loan

    def _release_new_idle(self, existing: frozenset[RecurrentState]) -> None:
        retained = []
        for bank in self._banks:
            if (
                bank not in existing
                and bank.convolution.exclusive_allocation
                and bank.delta.exclusive_allocation
            ):
                bank.close()
            else:
                retained.append(bank)
        self._banks = retained

    def reclaimable(self, states: tuple[QwenState, ...]) -> int:
        for state in states:
            state.check()
            if state.store is not self or state.pending is not None:
                raise ValueError("reclamation requires this store's reconciled states")
        tensors = tuple(
            tensor
            for state in states
            for bank in state.recurrent
            if bank is not None
            for tensor in (bank.convolution, bank.delta)
        )
        caches = tuple(
            tensor
            for bank in self._banks
            for tensor in (bank.convolution, bank.delta)
            if any(tensor.overlaps(owned) for owned in tensors)
        )
        kv = (
            0
            if self.pool is None
            else self.pool.reclaimable(tuple(span.run for state in states for span in state.kv))
        )
        return kv + reclaimable_bytes((*tensors, *caches))

    def release_idle(self) -> int:
        """Return unborrowed recurrent banks to the memory budget under pressure."""
        self.context.check()
        before = self.context.allocated_bytes
        retained = []
        for bank in self._banks:
            if bank.convolution.exclusive_allocation and bank.delta.exclusive_allocation:
                bank.close()
            else:
                retained.append(bank)
        self._banks = retained
        return before - self.context.allocated_bytes

    def create(self, checkpoint: QwenCheckpoint | None = None) -> QwenState:
        self.context.check()
        if self.closed:
            raise RuntimeError("Qwen state store is closed")
        if checkpoint is not None:
            if checkpoint.store is not self or checkpoint.closed:
                raise ValueError("checkpoint is not compatible with this state store")
            kv, recurrent = _fork_data(checkpoint.kv, checkpoint.recurrent)
            state = QwenState(self, checkpoint.position, kv, recurrent)
        else:
            with ExitStack() as cleanup:
                existing = frozenset(self._banks)
                cleanup.callback(lambda: self._release_new_idle(existing))
                recurrent = []
                for kind in self.geometry.layers:
                    value = self._recurrent(zero=True) if kind == MixerKind.RECURRENT else None
                    if value is not None:
                        cleanup.callback(value.close)
                    recurrent.append(value)
                state = QwenState(self, 0, (), tuple(recurrent))
                cleanup.pop_all()
        self._states.add(state)
        return state

    def close(self) -> None:
        self.context.check_thread()
        if not self.closed:
            for state in tuple(self._states):
                state.close()
            for checkpoint in tuple(self._checkpoints):
                checkpoint.close()
            for bank in self._banks:
                bank.close()
            self._banks.clear()
            if self.pool is not None:
                self.pool.close()
            self.closed = True
