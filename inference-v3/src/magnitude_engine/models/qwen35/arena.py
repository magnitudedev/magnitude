"""The bound program's one scratch allocation, shared by every operation.

There used to be three owners of intermediate storage: the program's slots, the
attention operation's private workspace, and the weight binding's projection
workspace. There is one now. A plan declares the regions it needs as a function
of shape; equal names share one allocation and the largest declaration wins, so
attention's scores are the same bytes in every layer, as they were before.

Growing a region invalidates the captured sequence, which is what ``generation``
reports to the program.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from enum import StrEnum

from magnitude_engine.kernels.precision import Precision
from magnitude_engine.operations.candidates import Scratch
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import DeviceContext, DType, Tensor, TensorSpec


class Slot(StrEnum):
    INPUT = "input"
    HIDDEN = "hidden"
    RESIDUAL = "residual"
    NORMALIZED = "normalized"
    READOUT = "readout"
    READOUT_INPUT = "readout_input"
    MIXER_OUTPUT = "mixer_output"
    MIXER_PROJECTIONS = "mixer_projections"
    QUERY_PREPARED = "query_prepared"
    KEY_PREPARED = "key_prepared"
    GATE_PREPARED = "gate_prepared"
    ATTENDED = "attended"
    ATTENTION_GATED = "attention_gated"
    RECURRENT_QUERY = "recurrent_query"
    RECURRENT_KEY = "recurrent_key"
    RECURRENT_VALUE = "recurrent_value"
    RECURRENT_BETA = "recurrent_beta"
    RECURRENT_DECAY = "recurrent_decay"
    RECURRENT_MIXED = "recurrent_mixed"
    RECURRENT_NORMALIZED = "recurrent_normalized"
    RECURRENT_GATED = "recurrent_gated"
    FEEDFORWARD_ACTIVATED = "feedforward_activated"


class Arena:
    def __init__(self, context: DeviceContext, precision: Precision):
        self.context, self.precision = context, precision
        self._buffers: dict[str, Tensor] = {}
        self.generation = 0

    def dtype(self, slot: Slot) -> DType:
        if slot in (Slot.HIDDEN, Slot.RESIDUAL, Slot.READOUT, Slot.READOUT_INPUT):
            return self.precision.residual
        if slot == Slot.RECURRENT_DECAY:
            return DType.F32
        if slot in (
            Slot.RECURRENT_QUERY,
            Slot.RECURRENT_KEY,
            Slot.RECURRENT_VALUE,
            Slot.RECURRENT_BETA,
            Slot.RECURRENT_MIXED,
        ):
            return self.precision.recurrent
        return self.precision.activation

    # ------------------------------------------------------------ reservation

    def reserve_slots(self, elements: dict[Slot, int]) -> None:
        self._grow(
            {
                slot.value: TensorSpec((count,), self.dtype(slot))
                for slot, count in elements.items()
            }
        )

    def reserve(self, regions: Iterable[Scratch]) -> None:
        """Declare operation scratch. Equal names share one allocation."""
        wanted: dict[str, TensorSpec] = {}
        for region in regions:
            current = wanted.get(region.name)
            if current is None or current.nbytes < region.spec.nbytes:
                wanted[region.name] = region.spec
        self._grow(wanted)

    def available(self, names: tuple[str, ...]) -> int:
        """What a plan may claim: the free budget plus what these regions hold."""
        held = sum(
            self._buffers[name].spec.nbytes
            for name in dict.fromkeys(names)
            if name in self._buffers
        )
        return self.context.budget_bytes - self.context.allocated_bytes + held

    def _grow(self, wanted: dict[str, TensorSpec]) -> None:
        self.context.check()
        with ExitStack() as cleanup:
            replacements: dict[str, Tensor] = {}
            for name, spec in wanted.items():
                current = self._buffers.get(name)
                if current is not None and current.spec.nbytes >= spec.nbytes:
                    continue
                # Regions are addressed as raw extents; a consumer views them
                # with whatever specification its schedule declared.
                buffer = self.context.allocate(TensorSpec((spec.nbytes,), DType.U8))
                cleanup.callback(buffer.close)
                replacements[name] = buffer
            for name, buffer in replacements.items():
                previous = self._buffers.get(name)
                self._buffers[name] = buffer
                if previous is not None:
                    previous.close()
            if replacements:
                self.generation += 1
            cleanup.pop_all()

    # ----------------------------------------------------------------- access

    def region(self, preparation: Preparation, name: str, spec: TensorSpec) -> Tensor:
        self.reserve((Scratch(name, spec),))
        return preparation.view(self._buffers[name], spec)

    def release_scratch(self) -> None:
        """Drop operation scratch, keeping the slots a forward always needs.

        A new operand layout discards the plan and the intermediates that plan
        sized; the program's own slots are sized by row count alone and survive.
        """
        slots = {slot.value for slot in Slot}
        for name in [name for name in self._buffers if name not in slots]:
            self._buffers.pop(name).close()

    def view(
        self, preparation: Preparation, slot: Slot, shape: tuple[int, ...], offset: int = 0
    ) -> Tensor:
        spec = TensorSpec(shape, self.dtype(slot))
        return preparation.view(self._buffers[slot.value], spec, offset)

    def acquire(self, slot: Slot, shape: tuple[int, ...]) -> Tensor:
        return self._buffers[slot.value].view(TensorSpec(shape, self.dtype(slot)))

    def close(self) -> None:
        for buffer in self._buffers.values():
            buffer.close()
        self._buffers.clear()
