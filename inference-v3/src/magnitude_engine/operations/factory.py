"""Weight-backed operation binding; residency is replaceable as one component."""

from abc import ABC, abstractmethod
from contextlib import ExitStack

from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.artifacts.weights import WeightDescriptor, WeightTransform
from magnitude_engine.operations.embedding import Embedding, EncodedEmbedding
from magnitude_engine.operations.gated import GatedLinear, ProjectionWorkspace
from magnitude_engine.operations.linear import EncodedLinear, Linear, ProjectionSchedule
from magnitude_engine.operations.parameters import DenseParameter, Parameter
from magnitude_engine.operations.projections import Projections, SeparateProjections
from magnitude_engine.operations.weights import ResidentWeight
from magnitude_engine.platform.execution import DeviceContext, DType


class WeightOperations(ABC):
    @property
    @abstractmethod
    def context(self) -> DeviceContext: ...

    @property
    @abstractmethod
    def artifact_identity(self) -> ArtifactIdentity: ...

    @abstractmethod
    def linear(self, weight: WeightDescriptor) -> Linear: ...

    @abstractmethod
    def gated_linear(
        self, gate: WeightDescriptor, up: WeightDescriptor, *, native_rounding: bool
    ) -> Linear: ...

    def projections(self, weights: tuple[WeightDescriptor, ...]) -> Projections:
        return SeparateProjections(tuple(self.linear(weight) for weight in weights))

    @abstractmethod
    def embedding(self, weight: WeightDescriptor) -> Embedding: ...

    @abstractmethod
    def parameter(self, weight: WeightDescriptor) -> Parameter: ...

    @abstractmethod
    def reserve_workspace(self, rows: int, dtype: DType) -> None: ...

    @abstractmethod
    def release_workspace(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class ResidentOperations(WeightOperations):
    def __init__(self, artifact: GGUFArtifact, context: DeviceContext, vector_rows: int = 1):
        if type(vector_rows) is not int or vector_rows <= 0:
            raise ValueError("vector row tile must be a positive integer")
        self.artifact, self._context = artifact, context
        self.vector_rows = vector_rows
        self._weights: dict[str, ResidentWeight] = {}
        self._linears: dict[str, Linear] = {}
        self._embeddings: dict[str, Embedding] = {}
        self._parameters: dict[str, Parameter] = {}
        self._cleanup = ExitStack()
        self._workspace = ProjectionWorkspace(context)
        self._cleanup.callback(self._workspace.close)
        self._gated: dict[tuple[WeightDescriptor, WeightDescriptor, bool], GatedLinear] = {}
        self._closed = False

    @property
    def context(self) -> DeviceContext:
        return self._context

    @property
    def artifact_identity(self) -> ArtifactIdentity:
        return self.artifact.identity

    def _weight(self, descriptor: WeightDescriptor) -> ResidentWeight:
        self.context.check()
        if self._closed:
            raise RuntimeError("operation binding is closed")
        if (
            self.artifact.directory.tensor(descriptor.name).shape != descriptor.shape
            or descriptor.transform != WeightTransform.IDENTITY
        ):
            raise ValueError("weight descriptor differs from the bound artifact")
        if descriptor.name not in self._weights:
            weight = ResidentWeight(self.artifact, self.context, descriptor.name)
            self._cleanup.callback(weight.close)
            self._weights[descriptor.name] = weight
        return self._weights[descriptor.name]

    def linear(self, weight: WeightDescriptor) -> Linear:
        resident = self._weight(weight)
        if weight.name not in self._linears:
            operation = EncodedLinear(
                resident,
                schedule=ProjectionSchedule.BASELINE,
                vector_rows=self.vector_rows,
            )
            self._cleanup.callback(operation.close)
            self._linears[weight.name] = operation
        return self._linears[weight.name]

    def gated_linear(
        self, gate: WeightDescriptor, up: WeightDescriptor, *, native_rounding: bool
    ) -> Linear:
        key = gate, up, native_rounding
        if key not in self._gated:
            operation = GatedLinear(
                self.projections((gate, up)),
                gate.shape[1],
                self._workspace,
                native_rounding=native_rounding,
            )
            self._cleanup.callback(operation.close)
            self._gated[key] = operation
        return self._gated[key]

    def embedding(self, weight: WeightDescriptor) -> Embedding:
        resident = self._weight(weight)
        if weight.name not in self._embeddings:
            operation = EncodedEmbedding(resident)
            self._cleanup.callback(operation.close)
            self._embeddings[weight.name] = operation
        return self._embeddings[weight.name]

    def parameter(self, weight: WeightDescriptor) -> Parameter:
        resident = self._weight(weight)
        if weight.name not in self._parameters:
            operation = DenseParameter(resident)
            self._cleanup.callback(operation.close)
            self._parameters[weight.name] = operation
        return self._parameters[weight.name]

    def reserve_workspace(self, rows: int, dtype: DType) -> None:
        for operation in self._gated.values():
            operation.reserve_workspace(rows, dtype)

    def release_workspace(self) -> None:
        self._workspace.close()

    def close(self) -> None:
        if not self._closed:
            self._cleanup.close()
            self._closed = True
