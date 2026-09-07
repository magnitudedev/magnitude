"""Expose physical page state through the model transaction contract."""

import mlx.core as mx

from ..inputs import ModelInputs
from .pages import KVCheckpoint, PageStore, SequencePages


class PagedTransaction:
    def __init__(self, state: SequencePages, count: int, *, committed_inputs: int = 0):
        if type(committed_inputs) is not int or not 0 <= committed_inputs <= count:
            raise ValueError("committed prefix is outside the state advance")
        self.committed_inputs = committed_inputs
        self.state = state
        self.base = state.length
        self.count = count
        self.reconciled = False
        state.reserve(self.base + count)

    def reconcile(self, accepted: int) -> ModelInputs | None:
        if self.reconciled or not self.committed_inputs <= accepted <= self.count:
            raise ValueError("invalid paged state reconciliation")
        self.state.commit(self.base + self.count)
        if accepted != self.count:
            self.state.trim(self.base + accepted)
        self.reconciled = True
        return None

    def finish(self, accepted: int) -> None:
        if not self.reconciled or self.state.length != self.base + accepted:
            raise RuntimeError("paged state did not reach the accepted boundary")

    def commit_all(self) -> None:
        self.reconcile(self.count)
        self.finish(self.count)

    def close(self) -> None:
        # Physical ownership stays with the sequence; pending execution owns its pin.
        pass


class PagedStateStore:
    def __init__(self, pages: PageStore):
        self.pages = pages

    def _check(self, state: SequencePages) -> None:
        if state.store is not self.pages or state.closed:
            raise ValueError("paged state is closed or belongs to another model")

    def create(self, checkpoint: KVCheckpoint | None = None) -> SequencePages:
        return self.pages.create(checkpoint)

    def reserve(self, state: SequencePages, input_capacity: int) -> None:
        self._check(state)
        state.reserve(state.length + input_capacity)

    def begin(
        self, state: SequencePages, inputs: ModelInputs, *, committed_inputs: int = 0
    ) -> PagedTransaction:
        self._check(state)
        return PagedTransaction(state, inputs.count, committed_inputs=committed_inputs)

    def arrays(self, state: SequencePages) -> tuple[mx.array, ...]:
        self._check(state)
        arrays = (*self.pages.arena.keys, *self.pages.arena.values)
        if state.tail is not None:
            arrays += state.tail.image.buffers
        return arrays

    def checkpoint(self, state: SequencePages) -> KVCheckpoint:
        self._check(state)
        return state.checkpoint()

    def release(self, state: SequencePages) -> None:
        self._check(state)
        state.close()
