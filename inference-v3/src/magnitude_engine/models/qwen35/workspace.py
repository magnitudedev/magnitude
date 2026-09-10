"""Shared ordered-execution scratch for Qwen's bound program."""

from contextlib import ExitStack
from enum import StrEnum

from magnitude_engine.numerics.policy import NumericalFamily
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


class Workspace:
    def __init__(
        self, context: DeviceContext, numerics: NumericalFamily = NumericalFamily.NATIVE_BF16
    ):
        self.context, self.numerics = context, numerics
        self._buffers: dict[Slot, Tensor] = {}
        self.generation = 0

    def dtype(self, slot: Slot) -> DType:
        if slot in (Slot.HIDDEN, Slot.RESIDUAL, Slot.READOUT, Slot.READOUT_INPUT):
            return self.numerics.residual
        if slot == Slot.RECURRENT_DECAY:
            return DType.F32
        if slot in (
            Slot.RECURRENT_QUERY,
            Slot.RECURRENT_KEY,
            Slot.RECURRENT_VALUE,
            Slot.RECURRENT_BETA,
            Slot.RECURRENT_MIXED,
        ):
            return self.numerics.recurrent_activation
        return self.numerics.activation

    def reserve(self, elements: dict[Slot, int]) -> None:
        self.context.check()
        with ExitStack() as cleanup:
            replacements = {}
            for slot, count in elements.items():
                current = self._buffers.get(slot)
                if current is None or current.spec.nbytes < count * self.dtype(slot).itemsize:
                    buffer = self.context.allocate(TensorSpec((count,), self.dtype(slot)))
                    cleanup.callback(buffer.close)
                    replacements[slot] = buffer
            for slot, buffer in replacements.items():
                previous = self._buffers.get(slot)
                self._buffers[slot] = buffer
                if previous is not None:
                    previous.close()
            if replacements:
                self.generation += 1
            cleanup.pop_all()

    def view(
        self, preparation: Preparation, slot: Slot, shape: tuple[int, ...], offset: int = 0
    ) -> Tensor:
        return preparation.view(self._buffers[slot], TensorSpec(shape, self.dtype(slot)), offset)

    def acquire(self, slot: Slot, shape: tuple[int, ...]) -> Tensor:
        return self._buffers[slot].view(TensorSpec(shape, self.dtype(slot)))

    def close(self) -> None:
        for buffer in self._buffers.values():
            buffer.close()
        self._buffers.clear()
