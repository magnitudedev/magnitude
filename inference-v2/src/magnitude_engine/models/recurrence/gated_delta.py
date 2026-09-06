"""Gated DeltaNet preparation, with recurrence implementation injected explicitly."""

from collections.abc import Callable
from dataclasses import dataclass, field

import mlx.core as mx

from ..execution import ExecutionScope
from ..state.recurrent import (
    RecurrentBoundaries,
    RecurrentLayout,
    RecurrentSlot,
    StateTensor,
    read_batch,
    write_batch,
)
from .contracts import DeltaRecurrence
from .graph import DeltaGraph
from .inputs import DeltaInputs


@dataclass(frozen=True)
class DeltaTransition:
    inputs: DeltaInputs
    convolution_input: mx.array
    initial: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    recurrence: DeltaRecurrence

    @property
    def length(self) -> int:
        return self.inputs.length

    def prefix(self, count: int) -> tuple[mx.array, ...]:
        if not 0 <= count <= self.length:
            raise ValueError("invalid gated-delta prefix")
        if count == 0:
            return self.initial
        if count == self.length:
            return self.values
        window = self.initial[0].shape[1]
        conv = mx.array(self.convolution_input[:, count : count + window])
        state = self.recurrence.reconcile(self.inputs, self.initial[1], count)
        return conv, state


@dataclass(frozen=True)
class GatedDelta:
    graph: DeltaGraph
    _advance: Callable[..., tuple[mx.array, ...]] = field(init=False, repr=False, compare=False)
    _trace: Callable[..., tuple[mx.array, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Capture only immutable tensor computation. State staging, leases and
        # acceptance remain outside tracing and run on every invocation.
        object.__setattr__(self, "_advance", mx.compile(self.graph.advance))
        object.__setattr__(self, "_trace", mx.compile(self.graph.__call__))

    def layout(self, dtype: mx.Dtype) -> RecurrentLayout:
        channels = (
            2 * self.graph.key_heads * self.graph.key_width
            + self.graph.value_heads * self.graph.value_width
        )
        itemsize = 4 if dtype == mx.float32 else 2
        # Raw conv input, normalized q/k/v, beta, float32 decay; state replacements
        # are reserved separately by the state transaction.
        trace_bytes = 2 * channels * itemsize + self.graph.value_heads * (itemsize + 4)
        return RecurrentLayout(
            (
                StateTensor((1, self.graph.window, channels), dtype),
                StateTensor(
                    (1, self.graph.value_heads, self.graph.value_width, self.graph.key_width),
                    mx.float32,
                ),
            ),
            trace_bytes,
        )

    def compute(self, hidden: mx.array, slot: RecurrentSlot, scope: ExecutionScope) -> mx.array:
        return self.compute_batch(hidden, (slot,), scope)

    def compute_batch(
        self,
        hidden: mx.array,
        slots: tuple[RecurrentSlot, ...],
        scope: ExecutionScope,
        *,
        committed_inputs: int = 0,
    ) -> mx.array:
        batch, count, _ = hidden.shape
        if not 0 <= committed_inputs <= count:
            raise ValueError("known recurrent prefix exceeds the input")
        if batch != len(slots) or not slots or len({id(slot) for slot in slots}) != batch:
            raise ValueError("gated-delta batch requires a distinct recurrent slot per row")
        conv, memory = read_batch(slots)
        if count == 1 or committed_inputs == count:
            output, new_conv, updated = self._advance(hidden, conv, memory)
            row_values = write_batch(slots, (new_conv, updated))
            for slot, values in zip(slots, row_values, strict=True):
                slot.stage(RecurrentBoundaries(slot.values, values, count))
            if count > 1:
                scope.submit_state(new_conv, updated)
            return output
        output, q, k, v, decay, beta, joined, updated = self._trace(hidden, conv, memory)
        prepared = DeltaInputs(q, k, v, decay, beta)
        new_conv = mx.array(joined[:, -self.graph.window :])
        row_states = write_batch(slots, (new_conv, updated))
        for row, slot in enumerate(slots):
            row_values = row_states[row]
            row_inputs = (
                prepared
                if batch == 1
                else DeltaInputs(
                    *(
                        mx.array(a[row : row + 1])
                        for a in (
                            prepared.queries,
                            prepared.keys,
                            prepared.values,
                            prepared.decay,
                            prepared.beta,
                        )
                    )
                )
            )
            row_joined = joined if batch == 1 else mx.array(joined[row : row + 1])
            slot.stage(
                DeltaTransition(
                    row_inputs,
                    row_joined,
                    slot.values,
                    row_values,
                    self.graph.recurrence,
                )
            )
            if batch > 1:
                # Interior-prefix repair outlives shared execution. Detach and
                # complete its retained inputs under that execution's obligation.
                scope.depend(
                    row_inputs.queries,
                    row_inputs.keys,
                    row_inputs.values,
                    row_inputs.decay,
                    row_inputs.beta,
                    row_joined,
                    *row_values,
                )
        if count > 1:
            scope.submit_state(new_conv, updated)
        return output
