"""Accepted sequence state and transactional updates over Ops resource ownership."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass

import ops


@dataclass(eq=False, slots=True)
class _Extent:
    start: int
    count: int
    claims: int = 1


class StateStore:
    """A bounded shared history arena and immutable per-sequence value versions.

    Arena capacity is shared by all descendants; no sequence owns a full-context
    slot. The arena is allocated lazily and can be released when its owners close.
    Ops owns physical allocations; this owner grants exclusive logical subranges.
    """

    def __init__(
        self,
        device: ops.DeviceRuntime,
        *,
        context_capacity: int,
        history_capacity: int,
        history_specs: tuple[ops.TensorSpec, ...],
        initial_values: Callable[[], tuple[ops.Resource, ...]],
    ):
        if (
            type(context_capacity) is not int
            or type(history_capacity) is not int
            or not 0 < context_capacity <= history_capacity
        ):
            raise ValueError("history capacity must fit a positive sequence context")
        if any(spec.rank < 1 or spec.shape[0] != history_capacity for spec in history_specs):
            raise ValueError("history specs must match their shared arena capacity")
        self.device, self.context_capacity = device, context_capacity
        self.history_capacity = history_capacity
        self._history_specs, self._initial_values = history_specs, initial_values
        self._history: tuple[ops.Resource, ...] = ()
        self._free = [(0, history_capacity)]
        self._states: set[SequenceState] = set()
        self._checkpoints: set[StateCheckpoint] = set()
        self.closed = False

    def check(self) -> None:
        self.device.check()
        if self.closed:
            raise RuntimeError("state store is closed")

    @property
    def history(self) -> tuple[ops.Resource, ...]:
        self.check()
        if not self._history and self._history_specs:
            with ExitStack() as cleanup:
                resources = []
                for spec in self._history_specs:
                    resource = self.device.allocate(spec)
                    cleanup.callback(resource.close)
                    resources.append(resource)
                self._history = tuple(resources)
                cleanup.pop_all()
        return self._history

    @property
    def occupied_rows(self) -> int:
        return self.history_capacity - sum(count for _, count in self._free)

    def _reserve(self, count: int) -> tuple[_Extent, ...]:
        if not self._history_specs:
            return ()
        if self._free and self._free[0][1] >= count:
            start, size = self._free[0]
            reserved = (_Extent(start, count),)
            if size == count:
                self._free.pop(0)
            else:
                self._free[0] = start + count, size - count
            return reserved
        available = sum(size for _, size in self._free)
        if available < count:
            total_bytes = sum(spec.storage_nbytes for spec in self._history_specs)
            required = (count * total_bytes + self.history_capacity - 1) // self.history_capacity
            available_bytes = available * total_bytes // self.history_capacity
            raise ops.CapacityError(required, available_bytes)
        # Construct the new free list first; allocation failure cannot partially
        # reserve history and leave no advance responsible for its release.
        remaining, reserved, free = count, [], []
        for start, size in self._free:
            taken = min(size, remaining)
            if taken:
                reserved.append(_Extent(start, taken))
                remaining -= taken
            if size > taken:
                free.append((start + taken, size - taken))
        result = tuple(reserved)
        self._free = free
        return result

    def _retain(self, extents: tuple[_Extent, ...]) -> tuple[_Extent, ...]:
        for extent in extents:
            extent.claims += 1
        return extents

    def _release(self, extents: tuple[_Extent, ...]) -> None:
        for extent in extents:
            extent.claims -= 1
            if extent.claims < 0:
                raise RuntimeError("history claim underflow")
            if extent.claims == 0:
                self._free.append((extent.start, extent.count))
        merged: list[tuple[int, int]] = []
        for start, count in sorted(self._free):
            if merged and sum(merged[-1]) == start:
                previous, length = merged.pop()
                merged.append((previous, length + count))
            else:
                merged.append((start, count))
        self._free = merged

    def create(self, checkpoint: StateCheckpoint | None = None) -> SequenceState:
        self.check()
        with ExitStack() as cleanup:
            if checkpoint is None:
                position, extents = 0, ()
                values = self._initial_values()
                for resource in values:
                    cleanup.callback(resource.close)
            else:
                if checkpoint.store is not self or checkpoint.closed:
                    raise ValueError("checkpoint is closed or belongs to another state store")
                position = checkpoint.position
                extents = self._retain(checkpoint._extents)
                cleanup.callback(self._release, extents)
                values = []
                for resource in checkpoint.values:
                    retained = resource.fork()
                    cleanup.callback(retained.close)
                    values.append(retained)
            result = SequenceState(self, position, extents, tuple(values))
            if checkpoint is not None:
                result.history_start = checkpoint.history_start
                result._retained_start = checkpoint._retained_start
            self._states.add(result)
            cleanup.pop_all()
            return result

    def reclaimable(self, states: Iterable[SequenceState]) -> int:
        selected = set(states)
        if any(state.store is not self or state.closed for state in selected):
            raise ValueError("reclamation requires live states from this store")
        values = tuple(value for state in selected for value in state.values)
        return ops.Resource.reclaimable_bytes(values)

    @property
    def idle(self) -> bool:
        return not self._states and not self._checkpoints

    def release_idle(self) -> int:
        self.check()
        if not self.idle:
            return 0
        released = ops.Resource.reclaimable_bytes(self._history)
        for resource in self._history:
            resource.close()
        self._history = ()
        return released

    def close(self) -> None:
        if not self.closed:
            for state in tuple(self._states):
                state.close()
            for checkpoint in tuple(self._checkpoints):
                checkpoint.close()
            self.release_idle()
            self.closed = True


class SequenceState:
    def __init__(
        self,
        store: StateStore,
        position: int,
        extents: tuple[_Extent, ...],
        values: tuple[ops.Resource, ...],
    ):
        self.store, self.position, self._extents, self.values = store, position, extents, values
        self.history_start = self._retained_start = 0
        self.expected_end = position
        self.pending: StateAdvance | None = None
        self.closed = False

    def check(self) -> None:
        self.store.check()
        if self.closed:
            raise RuntimeError("sequence state is closed")

    def anticipate(self, position: int) -> None:
        self.check()
        if type(position) is not int or not 0 <= position <= self.store.context_capacity:
            raise ValueError("anticipated position exceeds context capacity")
        self.expected_end = max(self.expected_end, position)

    @property
    def history_ranges(self) -> tuple[tuple[int, int], ...]:
        self.check()
        spans: list[tuple[int, int]] = []
        skip = self.history_start - self._retained_start
        for extent in self._extents:
            omitted = min(skip, extent.count)
            skip -= omitted
            start, count = extent.start + omitted, extent.count - omitted
            if not count:
                continue
            if spans and sum(spans[-1]) == start:
                previous, length = spans.pop()
                spans.append((previous, length + count))
            else:
                spans.append((start, count))
        return tuple(spans)

    def trim_history(self, before: int) -> None:
        """Stop exposing entries before a logical position for this owner only.

        Whole unclaimed extents are recycled. A partially visible extent remains
        charged until its final owner releases it; trimming never copies bytes.
        Values and the accepted sequence position are unchanged.
        """
        self.check()
        if self.pending is not None:
            raise RuntimeError("history trimming requires reconciled state")
        if type(before) is not int or not self.history_start <= before <= self.position:
            raise ValueError("history trim must lie within accepted logical positions")
        retained, count = self._retained_start, 0
        for extent in self._extents:
            if retained + extent.count > before:
                break
            retained += extent.count
            count += 1
        remaining = self._extents[count:]
        self.store._release(self._extents[:count])
        self._extents = remaining
        self._retained_start, self.history_start = retained, before

    def begin(self, count: int) -> StateAdvance:
        self.check()
        if self.pending is not None:
            raise RuntimeError("state has an unresolved advance")
        if type(count) is not int or not 0 < count <= self.store.context_capacity - self.position:
            raise ValueError("advance exceeds context capacity")
        result = StateAdvance(self, count, self.store._reserve(count))
        self.pending = result
        return result

    def checkpoint(self) -> StateCheckpoint:
        self.check()
        if self.pending is not None:
            raise RuntimeError("checkpoint requires reconciled state")
        with ExitStack() as cleanup:
            extents = self.store._retain(self._extents)
            cleanup.callback(self.store._release, extents)
            values = []
            for value in self.values:
                retained = value.fork()
                cleanup.callback(retained.close)
                values.append(retained)
            result = StateCheckpoint(self.store, self.position, extents, tuple(values))
            result.history_start, result._retained_start = self.history_start, self._retained_start
            self.store._checkpoints.add(result)
            cleanup.pop_all()
            return result

    def close(self) -> None:
        if not self.closed:
            if self.pending is not None:
                self.pending.abort()
            for value in self.values:
                value.close()
            self.store._release(self._extents)
            self.store._states.discard(self)
            self.closed = True


class StateCheckpoint:
    def __init__(
        self,
        store: StateStore,
        position: int,
        extents: tuple[_Extent, ...],
        values: tuple[ops.Resource, ...],
    ):
        self.store, self.position, self._extents, self.values = store, position, extents, values
        self.history_start = self._retained_start = 0
        self.closed = False

    def fork(self) -> SequenceState:
        return self.store.create(self)

    def close(self) -> None:
        if not self.closed:
            for value in self.values:
                value.close()
            self.store._release(self._extents)
            self.store._checkpoints.discard(self)
            self.closed = True


class StateAdvance:
    def __init__(self, state: SequenceState, count: int, extents: tuple[_Extent, ...]):
        self.state, self.position, self.count = state, state.position, count
        self._extents = extents
        self.completion: ops.Completion | None = None
        self.following: tuple[ops.Resource, ...] | None = None
        self.closed = False

    @property
    def destinations(self) -> tuple[int, ...]:
        if self.closed:
            raise RuntimeError("advance is closed")
        return tuple(
            row
            for extent in self._extents
            for row in range(extent.start, extent.start + extent.count)
        )

    def submitted(self, completion: ops.Completion, following: tuple[ops.Resource, ...]) -> None:
        if self.closed or self.completion is not None or self.state.pending is not self:
            raise RuntimeError("advance is not awaiting submission")
        device = self.state.store.device
        if completion.device is not device:
            raise ValueError("completion belongs to another device")
        handles = set(following)
        if len(handles) != len(following) or not handles.isdisjoint(self.state.values):
            raise ValueError("successor components require independently owned resource handles")
        if len(following) != len(self.state.values) or any(
            value.device is not device or (value.spec is not old.spec and value.spec != old.spec)
            for value, old in zip(following, self.state.values, strict=True)
        ):
            raise ValueError("successor values differ from the accepted component schema")
        self.completion, self.following = completion, following

    def commit(self) -> None:
        self.state.check()
        if self.closed or self.state.pending is not self or self.state.position != self.position:
            raise RuntimeError("advance is no longer current")
        if self.completion is None or self.following is None:
            raise RuntimeError("advance has not been submitted")
        self.completion.wait()
        # Combine only exclusively owned adjacent extents. A checkpoint's
        # boundary must never grow when a descendant appends more tokens.
        previous = self.state._extents
        following = self._extents
        if (
            previous
            and following
            and previous[-1].claims == 1
            and previous[-1].start + previous[-1].count == following[0].start
        ):
            extents = previous + following[1:] if len(following) > 1 else previous
            previous[-1].count += following[0].count
            following[0].claims = 0
        else:
            extents = previous + following
        values = self.state.values
        self.state._extents, self.state.values = extents, self.following
        self.state.position += self.count
        self.state.pending = None
        self.following, self._extents, self.closed = None, (), True
        for value in values:
            value.close()

    def abort(self) -> None:
        if not self.closed:
            if self.completion is not None:
                # Completion pins preserve allocations, not exclusive ranges.
                # Do not recycle destinations while a submitted kernel can write.
                self.completion.wait()
            if self.following is not None:
                for value in self.following:
                    value.close()
            self.state.store._release(self._extents)
            self._extents = ()
            if self.state.pending is self:
                self.state.pending = None
            self.closed = True
