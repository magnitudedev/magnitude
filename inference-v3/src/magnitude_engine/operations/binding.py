"""Weight-backed operations over one residency, for any container.

There is one binding class. Which schedules its operations use follows from the
representation residency chose, so a new weight format adds a format module and
nothing here. It owns no scratch: that belongs to the bound program's arena.
"""

from __future__ import annotations

from contextlib import ExitStack

from magnitude_engine.kernels.precision import Precision
from magnitude_engine.operations.candidates import ScratchArena
from magnitude_engine.operations.embedding import Embedding, ResidentEmbedding
from magnitude_engine.operations.gated import GatedLinear
from magnitude_engine.operations.linear import Linear, ResidentLinear
from magnitude_engine.operations.parameters import Parameter, ResidentParameter
from magnitude_engine.operations.projections import (
    Projections,
    ResidentProjections,
    SeparateProjections,
)
from magnitude_engine.platform.execution import DeviceContext, DType
from magnitude_engine.weights.descriptor import WeightDescriptor
from magnitude_engine.weights.identity import ArtifactIdentity
from magnitude_engine.weights.residency import Weights


class Operations:
    def __init__(self, weights: Weights, precision: Precision, arena: ScratchArena):
        self.weights, self.precision, self.arena = weights, precision, arena
        self._linears: dict[str, Linear] = {}
        self._projections: dict[tuple[str, ...], Projections] = {}
        self._embeddings: dict[str, Embedding] = {}
        self._parameters: dict[str, Parameter] = {}
        self._gated: dict[tuple[str, str], GatedLinear] = {}
        self._cleanup = ExitStack()
        self._closed = False

    @property
    def context(self) -> DeviceContext:
        return self.weights.context

    @property
    def artifact_identity(self) -> ArtifactIdentity:
        return self.weights.identity

    def _check(self) -> None:
        self.context.check()
        if self._closed:
            raise RuntimeError("operation binding is closed")

    def _own[T](self, value: T) -> T:
        self._cleanup.callback(value.close)  # type: ignore[attr-defined]
        return value

    def projections(self, weights: tuple[WeightDescriptor, ...]) -> Projections:
        self._check()
        key = tuple(weight.name for weight in weights)
        if key not in self._projections:
            group = self.weights.group(weights) if len(weights) > 1 else None
            if group is not None:
                operation: Projections = ResidentProjections(group, self.precision, self.arena)
            elif len(weights) == 1:
                operation = self._resident_projections(weights[0])
            else:
                operation = SeparateProjections(
                    tuple(self._resident_projections(weight) for weight in weights)
                )
            self._projections[key] = self._own(operation)
        return self._projections[key]

    def _resident_projections(self, weight: WeightDescriptor) -> ResidentProjections:
        # A weight already resident as part of a group keeps that residency; a
        # tied readout and embedding are the same allocation, viewed twice.
        resident = self.weights.existing(weight)
        if resident is None:
            group = self.weights.group((weight,))
            resident = group if group is not None else self.weights.resident(weight)
        return ResidentProjections(resident, self.precision, self.arena)

    def linear(self, weight: WeightDescriptor) -> Linear:
        self._check()
        if weight.name not in self._linears:
            projection = self.projections((weight,))
            assert isinstance(projection, ResidentProjections)
            self._linears[weight.name] = self._own(ResidentLinear(projection))
        return self._linears[weight.name]

    def gated_linear(self, gate: WeightDescriptor, up: WeightDescriptor) -> Linear:
        self._check()
        key = gate.name, up.name
        if key not in self._gated:
            self._gated[key] = self._own(
                GatedLinear(
                    self.projections((gate, up)), gate.shape[1], self.precision, self.arena
                )
            )
        return self._gated[key]

    def embedding(self, weight: WeightDescriptor) -> Embedding:
        self._check()
        if weight.name not in self._embeddings:
            self._embeddings[weight.name] = self._own(
                ResidentEmbedding(self.weights.resident(weight), self.precision)
            )
        return self._embeddings[weight.name]

    def parameter(self, weight: WeightDescriptor) -> Parameter:
        self._check()
        if weight.name not in self._parameters:
            self._parameters[weight.name] = self._own(
                ResidentParameter(self.weights.resident(weight))
            )
        return self._parameters[weight.name]

    def reserve(self, rows: int, dtype: DType) -> None:
        """Declare the scratch every bound operation needs at this row count."""
        for operation in self._gated.values():
            operation.reserve(rows, dtype, dtype)

    def close(self) -> None:
        if not self._closed:
            self._cleanup.close()
            self._closed = True
