"""Logical Qwen continuation state over Ops resources."""

from __future__ import annotations

import struct
from contextlib import ExitStack
from dataclasses import dataclass

import ops
from engine.models.qwen35.description import Geometry, MixerKind


@dataclass(frozen=True, slots=True)
class RecurrentState:
    convolution: ops.Resource
    delta: ops.Resource

    def fork(self) -> RecurrentState:
        return RecurrentState(self.convolution.fork(), self.delta.fork())

    def close(self) -> None:
        self.convolution.close()
        self.delta.close()


class QwenState:
    def __init__(self, store: QwenStateStore, slot: int, position: int, recurrent):
        self.store, self.slot, self.position = store, slot, position
        self.recurrent = tuple(recurrent)
        self.expected_end = position
        self.pending: QwenAdvance | None = None
        self.closed = False

    def check(self) -> None:
        self.store.device.check()
        if self.closed or self.store.closed:
            raise RuntimeError("Qwen state is closed")

    def anticipate(self, position: int) -> None:
        self.check()
        if type(position) is not int or not 0 <= position <= self.store.context_capacity:
            raise ValueError("anticipated input exceeds the model context")
        self.expected_end = max(self.expected_end, position)

    def begin(self, count: int) -> QwenAdvance:
        self.check()
        if self.pending is not None:
            raise RuntimeError("state already has an unresolved advance")
        if (
            type(count) is not int
            or count <= 0
            or self.position + count > self.store.context_capacity
        ):
            raise ValueError("input advance exceeds the model context")
        result = QwenAdvance(self, count)
        self.pending = result
        return result

    def checkpoint(self) -> QwenCheckpoint:
        self.check()
        if self.pending is not None:
            raise RuntimeError("cannot checkpoint an unresolved advance")
        with ExitStack() as cleanup:
            self.store._retain_slot(self.slot)
            cleanup.callback(self.store._release_slot, self.slot)
            recurrent = []
            for value in self.recurrent:
                retained = value.fork()
                cleanup.callback(retained.close)
                recurrent.append(retained)
            checkpoint = QwenCheckpoint(self.store, self.slot, self.position, tuple(recurrent))
            self.store._checkpoints.add(checkpoint)
            cleanup.pop_all()
            return checkpoint

    def close(self) -> None:
        if not self.closed:
            if self.pending is not None:
                self.pending.abort()
            for value in self.recurrent:
                value.close()
            self.store._release_slot(self.slot)
            self.closed = True
            self.store._states.discard(self)


class QwenCheckpoint:
    def __init__(self, store, slot, position, recurrent):
        self.store, self.slot, self.position, self.recurrent = store, slot, position, recurrent
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            for value in self.recurrent:
                value.close()
            self.store._release_slot(self.slot)
            self.closed = True
            self.store._checkpoints.discard(self)


class QwenAdvance:
    def __init__(self, state: QwenState, count: int):
        self.state, self.count, self.position = state, count, state.position
        self.following: tuple[RecurrentState, ...] | None = None
        self.completion: ops.Completion | None = None
        self.closed = False

    def submitted(self, completion: ops.Completion, following) -> None:
        if self.closed or self.completion is not None or self.state.pending is not self:
            raise RuntimeError("advance is not awaiting submission")
        if completion.device is not self.state.store.device:
            raise ValueError("advance was submitted on another device")
        self.completion, self.following = completion, tuple(following)

    def commit(self) -> None:
        self.state.check()
        if self.closed or self.state.pending is not self or self.state.position != self.position:
            raise RuntimeError("advance is no longer current")
        if self.completion is None or self.following is None:
            raise RuntimeError("state commit requires a submitted execution")
        self.completion.wait()
        previous = self.state.recurrent
        self.state.recurrent = self.following
        self.following = None
        for value in previous:
            value.close()
        self.state.position += self.count
        self.state.pending = None
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            if self.following is not None:
                for value in self.following:
                    value.close()
            if self.state.pending is self:
                self.state.pending = None
            self.closed = True


class QwenStateStore:
    def __init__(
        self,
        device: ops.DeviceRuntime,
        geometry: Geometry,
        slots: int = 8,
        context_capacity: int | None = None,
        kv_representation: ops.KVRepresentation | None = None,
    ):
        if slots <= 0:
            raise ValueError("Qwen state slot count must be positive")
        if context_capacity is None:
            context_capacity = geometry.context_limit
        if (
            type(context_capacity) is not int
            or context_capacity <= 0
            or context_capacity > geometry.context_limit
        ):
            raise ValueError("Qwen state context capacity must fit the model context")
        self.device, self.geometry, self.slots = device, geometry, slots
        self.context_capacity = context_capacity
        self._claims = [0] * slots
        self._states: set[QwenState] = set()
        self._checkpoints: set[QwenCheckpoint] = set()
        attention = sum(kind == MixerKind.ATTENTION for kind in geometry.layers)
        representation = kv_representation or ops.default_kv_representation(
            geometry.attention_width, geometry.attention_width)
        with ExitStack() as cleanup:
            caches = []
            for _ in range(attention):
                cache = device.allocate(
                    ops.kv_state_spec(slots * context_capacity, geometry.kv_heads,
                                      geometry.activation_dtype, representation)
                )
                cleanup.callback(cache.close)
                caches.append(cache)
            self.attention = tuple(caches)
            cleanup.pop_all()
        self._copy_programs: dict[int, ops.CompiledFunction] = {}
        self.closed = False

    def _slot(self) -> int:
        try:
            slot = self._claims.index(0)
        except ValueError as error:
            raise ops.CapacityError(1, 0) from error
        self._claims[slot] = 1
        return slot

    def _retain_slot(self, slot: int) -> None:
        self._claims[slot] += 1

    def _release_slot(self, slot: int) -> None:
        self._claims[slot] -= 1
        if self._claims[slot] < 0:
            raise RuntimeError("Qwen state slot claim underflow")

    def _zero_recurrent(self):
        g = self.geometry
        with ExitStack() as cleanup:
            result = []
            for kind in g.layers:
                if kind != MixerKind.RECURRENT:
                    continue
                convolution = ops.TensorSpec(
                    (1, g.recurrent_channels, g.convolution_width - 1),
                    g.activation_dtype,
                )
                delta = ops.TensorSpec(
                    (1, g.recurrent_value_heads, g.recurrent_width, g.recurrent_width),
                    ops.DType.F32,
                )
                convolution_resource = self.device.upload(
                    convolution, bytes(convolution.storage_nbytes)
                )
                cleanup.callback(convolution_resource.close)
                delta_resource = self.device.upload(delta, bytes(delta.storage_nbytes))
                cleanup.callback(delta_resource.close)
                state = RecurrentState(convolution_resource, delta_resource)
                cleanup.callback(state.close)
                result.append(state)
            cleanup.pop_all()
            return tuple(result)

    def create(self, checkpoint: QwenCheckpoint | None = None) -> QwenState:
        self.device.check()
        if self.closed:
            raise RuntimeError("Qwen state store is closed")
        slot = self._slot()
        with ExitStack() as cleanup:
            cleanup.callback(self._release_slot, slot)
            if checkpoint is None:
                position, recurrent = 0, self._zero_recurrent()
            else:
                if checkpoint.store is not self or checkpoint.closed:
                    raise ValueError("checkpoint is not compatible with this state store")
                position = checkpoint.position
                recurrent = tuple(value.fork() for value in checkpoint.recurrent)
            for value in recurrent:
                cleanup.callback(value.close)
            if checkpoint is not None:
                self._copy_slot(checkpoint.slot, slot, position)
            state = QwenState(self, slot, position, recurrent)
            self._states.add(state)
            cleanup.pop_all()
            return state

    def _copy_slot(self, source: int, target: int, count: int) -> None:
        if not count or not self.attention:
            return
        range_spec = ops.TensorSpec((1, 3), ops.DType.I32)
        program = self._copy_programs.get(count)
        if program is None:
            kwargs = {
                f"cache.{index}": ops.Argument(cache.spec, f"cache.{index}", ops.ValueKind.RESOURCE)
                for index, cache in enumerate(self.attention)
            }

            def copy(ranges, **resources):
                return tuple(
                    ops.kv_copy(resources[f"cache.{index}"], ranges, max_count=count)
                    for index in range(len(self.attention))
                )

            program = ops.compile(
                copy,
                signature=ops.Signature((ops.Argument(range_spec, "ranges"),), kwargs),
                device=self.device,
                constants={},
                options=ops.CompileOptions(mode="state-copy"),
            )
            self._copy_programs[count] = program
        base = self.context_capacity
        ranges = self.device.upload(
            range_spec, struct.pack("=iii", source * base, target * base, count)
        )
        try:
            execution = program.submit(
                ranges,
                resources={f"cache.{i}": cache for i, cache in enumerate(self.attention)},
            )
            execution.completion.wait()
            for output in execution.outputs:
                output.close()
        finally:
            ranges.close()

    def reclaimable(self, states) -> int:
        return sum(
            resource.allocated_bytes
            for state in states
            for recurrent in state.recurrent
            for resource in (recurrent.convolution, recurrent.delta)
        )

    def release_idle(self) -> int:
        return 0

    def close(self) -> None:
        if not self.closed:
            for state in tuple(self._states):
                state.close()
            for checkpoint in tuple(self._checkpoints):
                checkpoint.close()
            for program in self._copy_programs.values():
                program.close()
            for cache in self.attention:
                cache.close()
            self.closed = True
