"""One committed boundary across paged attention and recurrent state components."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine import components as c
from magnitude_engine.components import component
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.retention import RetainedStorage

from ..inputs import ModelInputs
from .pages import KVCheckpoint, PageStore, SequencePages
from .recurrent import RecurrentImage, RecurrentLayout, RecurrentPin, RecurrentRow, RecurrentSlot


class HybridState:
    def __init__(
        self,
        store: HybridStateStore,
        pages: SequencePages,
        recurrent: RecurrentRow,
    ):
        self.store = store
        self.pages = pages
        self.recurrent = recurrent
        self.slots = tuple(RecurrentSlot(recurrent, i) for i in range(len(store.layouts)))
        self.active: HybridTransaction | None = None
        self.closed = False

    @property
    def position(self) -> int:
        return self.pages.length


class HybridCheckpoint:
    def __init__(
        self,
        store: HybridStateStore,
        pages: KVCheckpoint,
        recurrent: RecurrentRow,
    ):
        self.store = store
        self.pages = pages
        self.recurrent = recurrent
        self.length = pages.length
        self.closed = False

    @property
    def reclaimable(self) -> bool:
        return self.pages.reclaimable

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        image = self.recurrent.image
        return (*self.pages.retained_storage(), RetainedStorage(image, image.reservation.size))

    def close(self) -> None:
        if self.closed:
            return
        self.pages.close()
        self.recurrent.close()
        self.closed = True


class HybridTransaction:
    def __init__(
        self,
        state: HybridState,
        count: int,
        destination: RecurrentRow,
        *,
        committed_inputs: int = 0,
    ):
        if type(committed_inputs) is not int or not 0 <= committed_inputs <= count:
            raise ValueError("committed prefix is outside the state advance")
        self.committed_inputs = committed_inputs
        self.state = state
        self.base = state.position
        self.count = count
        self.reconciled = False
        self.closed = False
        # Initial/final images already cover one-input and fully causal advances.
        # Only wider tentative work retains a trace for interior-prefix repair.
        trace_count = count if count > 1 and committed_inputs < count else 0
        self.trace_reservation = state.store.budget.reserve(
            "recurrent-advance",
            trace_count * sum(s.layout.trace_bytes_per_token for s in state.slots),
        )
        try:
            state.pages.reserve(self.base + count)
        except BaseException:
            self.trace_reservation.close()
            raise
        self.initial = state.recurrent.acquire()
        self.destination = destination
        self.repaired: RecurrentRow | None = None
        self._pins: tuple[RecurrentPin, ...] = ()
        for slot in state.slots:
            slot.destination = destination
        state.active = self

    def reconcile(self, accepted: int) -> ModelInputs | None:
        if self.reconciled or not self.committed_inputs <= accepted <= self.count:
            raise ValueError("invalid hybrid state reconciliation")
        self._validate_transitions()
        if accepted == self.count:
            result = self.destination
        elif accepted == 0:
            result = self.initial
        else:
            result = self.repaired = RecurrentImage(
                self.state.store.layouts, 1, self.state.store.budget
            ).acquire(0)
            for index, slot in enumerate(self.state.slots):
                assert slot.pending is not None
                result.image.write(index, slot.pending.prefix(accepted))
        # Partial recurrence replay completes before either boundary is published.
        mx.eval(*(a for i in range(len(self.state.slots)) for a in result.values(i)))
        self._publish(accepted, result)
        return None

    def _validate_transitions(self) -> None:
        for slot in self.state.slots:
            transition = slot.pending
            if transition is None or transition.length != self.count:
                raise RuntimeError("recurrent layer did not advance the complete input")

    def _publish(self, accepted: int, result: RecurrentRow) -> None:
        self.state.pages.commit(self.base + self.count)
        if accepted != self.count:
            self.state.pages.trim(self.base + accepted)
        previous = self.state.recurrent
        self.state.recurrent = result.acquire()
        for slot in self.state.slots:
            slot.source = self.state.recurrent
        previous.close()
        self.reconciled = True

    def commit_all(self) -> None:
        if self.closed or self.reconciled:
            raise RuntimeError("hybrid transaction is already resolved")
        self._validate_transitions()
        self._publish(self.count, self.destination)
        self.finish(self.count)
        self._detach()
        # Full commitment cannot read the old state again. Device consumers own
        # their submitted buffers (or their still-lazy output graph), while these
        # pins preserve capacity accounting until the execution retires us.
        self._pins = (self.initial.image.pin(), self.destination.image.pin())
        self.initial.close()
        self.destination.close()

    def finish(self, accepted: int) -> None:
        if not self.reconciled or self.state.position != self.base + accepted:
            raise RuntimeError("hybrid state did not reach its accepted boundary")

    def _detach(self) -> None:
        if self.state.active is self:
            for slot in self.state.slots:
                slot.pending = None
                slot.destination = None
            self.state.active = None

    def close(self) -> None:
        if not self.closed:
            self._detach()
            self.initial.close()
            self.destination.close()
            for pin in self._pins:
                pin.close()
            self._pins = ()
            if self.repaired is not None:
                self.repaired.close()
            self.trace_reservation.close()
            self.closed = True


@dataclass
class _Preparation:
    states: tuple[HybridState, ...]
    width: int
    image: RecurrentImage | None = None
    begun: int = 0


@component(c.HYBRID_STATE, source=c.Source.MAG, variant="HYBRID")
class HybridStateStore:
    def __init__(
        self, pages: PageStore, layouts: tuple[RecurrentLayout, ...], budget: MemoryBudget
    ):
        self.pages = pages
        self.layouts = layouts
        self.budget = budget
        self._prepared: _Preparation | None = None

    def can_batch(self, states: tuple[HybridState, ...]) -> bool:
        return bool(states) and all(state.store is self and not state.closed for state in states)

    def repair_group(self, state: HybridState) -> object:
        self._check(state)
        return self

    def prepare_batch(self, states: tuple[HybridState, ...], width: int) -> None:
        if width < 1 or not self.can_batch(states) or len(set(states)) != len(states):
            raise ValueError("invalid hybrid state batch")
        if any(state.active is not None for state in states):
            raise RuntimeError("batch preparation requires idle hybrid states")
        # Preparation describes the next transaction group. It owns no allocation;
        # begun transactions own the destination, including partial admission failure.
        self._prepared = _Preparation(states, width)

    def _destination(self, state: HybridState, count: int) -> RecurrentRow:
        prepared = self._prepared
        if prepared is None:
            return RecurrentImage(self.layouts, 1, self.budget).acquire(0)
        if prepared.width != count or prepared.states[prepared.begun] is not state:
            raise ValueError("hybrid transactions differ from the prepared batch")
        if prepared.image is None:
            prepared.image = RecurrentImage(self.layouts, len(prepared.states), self.budget)
        row = prepared.image.acquire(prepared.begun)
        prepared.begun += 1
        if prepared.begun == len(prepared.states):
            self._prepared = None
        return row

    def _copy(self, source: RecurrentRow) -> RecurrentRow:
        row = RecurrentImage(self.layouts, 1, self.budget).acquire(0)
        try:
            for i in range(len(self.layouts)):
                row.image.write(i, tuple(mx.array(a) for a in source.values(i)))
            mx.eval(*(a for i in range(len(self.layouts)) for a in row.values(i)))
            return row
        except BaseException:
            row.close()
            raise

    def _check(self, state: HybridState) -> None:
        if state.store is not self or state.closed:
            raise ValueError("hybrid state is closed or belongs to another model")

    def create(self, checkpoint: HybridCheckpoint | None = None) -> HybridState:
        if checkpoint is not None and (checkpoint.closed or checkpoint.store is not self):
            raise ValueError("hybrid checkpoint is closed or belongs to another model")
        recurrent = (
            self._copy(checkpoint.recurrent)
            if checkpoint is not None
            else RecurrentImage(self.layouts, 1, self.budget).acquire(0)
        )
        pages = None
        try:
            if checkpoint is None:
                for i, layout in enumerate(self.layouts):
                    recurrent.image.write(
                        i, tuple(mx.zeros(t.shape, dtype=t.dtype) for t in layout.tensors)
                    )
                mx.eval(*(a for i in range(len(self.layouts)) for a in recurrent.values(i)))
            pages = self.pages.create(None if checkpoint is None else checkpoint.pages)
            return HybridState(self, pages, recurrent)
        except BaseException:
            if pages is not None:
                pages.close()
            recurrent.close()
            raise

    def reserve(self, state: HybridState, input_capacity: int) -> None:
        self._check(state)
        if state.active:
            raise RuntimeError("capacity preparation requires reconciled hybrid state")
        state.pages.reserve(state.position + input_capacity)

    def begin(
        self, state: HybridState, inputs: ModelInputs, *, committed_inputs: int = 0
    ) -> HybridTransaction:
        self._check(state)
        if state.active:
            raise RuntimeError("hybrid state already has a pending advance")
        destination = None
        try:
            destination = self._destination(state, inputs.count)
            return HybridTransaction(
                state, inputs.count, destination, committed_inputs=committed_inputs
            )
        except BaseException:
            self._prepared = None
            if destination is not None:
                destination.close()
            raise

    def arrays(self, state: HybridState) -> tuple[mx.array, ...]:
        self._check(state)
        arrays = [*self.pages.arena.keys, *self.pages.arena.values]
        for slot in state.slots:
            arrays.extend(slot.values if slot.pending is None else slot.pending.values)
        return tuple(arrays)

    def checkpoint(self, state: HybridState) -> HybridCheckpoint:
        self._check(state)
        if state.active:
            raise RuntimeError("hybrid checkpoint requires reconciled state")
        recurrent = self._copy(state.recurrent)
        try:
            pages = state.pages.checkpoint()
        except BaseException:
            recurrent.close()
            raise
        return HybridCheckpoint(self, pages, recurrent)

    def release(self, state: HybridState) -> None:
        self._check(state)
        if state.active:
            raise RuntimeError("complete hybrid work before releasing its state")
        state.pages.close()
        state.recurrent.close()
        state.closed = True
