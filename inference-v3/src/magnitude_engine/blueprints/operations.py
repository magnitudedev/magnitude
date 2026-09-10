from magnitude_engine.artifacts.mlx import MLXArtifact
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.numerics.encoded_layout import EncodedLayout
from magnitude_engine.numerics.semantics import HeadMapping
from magnitude_engine.operations.attention import CausalAttention as Attention
from magnitude_engine.operations.embedding import Embedding
from magnitude_engine.operations.factory import WeightOperations
from magnitude_engine.operations.linear import Linear, ProjectionSchedule
from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.operations.recurrent import DeltaRecurrence as Recurrence
from magnitude_engine.operations.sampling import SampleSelector as Selector
from magnitude_engine.operations.weights import ResidentWeight as Weight
from magnitude_engine.platform.execution import DeviceContext

__all__ = [
    "EncodedLinear",
    "EncodedEmbedding",
    "ResidentWeight",
    "DenseParameter",
    "ResidentOperations",
    "MLXOperations",
    "SampleSelector",
    "DeltaRecurrence",
    "CausalAttention",
]


@blueprint
class CausalAttention(Blueprint[Attention]):
    context: Blueprint[DeviceContext]
    heads: int
    kv_heads: int
    width: int

    @staticmethod
    def implementation():
        return Attention


@blueprint
class DeltaRecurrence(Blueprint[Recurrence]):
    context: Blueprint[DeviceContext]
    key_heads: int
    value_heads: int
    width: int
    mapping: HeadMapping

    @staticmethod
    def implementation():
        return Recurrence


@blueprint
class SampleSelector(Blueprint[Selector]):
    context: Blueprint[DeviceContext]
    tile: int = 1024

    @staticmethod
    def implementation():
        return Selector


@blueprint
class ResidentWeight(Blueprint[Weight]):
    artifact: Blueprint[GGUFArtifact]
    context: Blueprint[DeviceContext]
    tensor_name: str
    layout: EncodedLayout | None = None

    @staticmethod
    def implementation():
        return Weight


@blueprint
class ResidentOperations(Blueprint[WeightOperations]):
    artifact: Blueprint[GGUFArtifact]
    context: Blueprint[DeviceContext]
    vector_rows: int = 1

    @staticmethod
    def implementation():
        from magnitude_engine.operations.factory import ResidentOperations

        return ResidentOperations


@blueprint
class EncodedLinear(Blueprint[Linear]):
    weight: Blueprint[Weight]
    output_tile: int = 4
    reduction_lanes: int = 32
    schedule: ProjectionSchedule = ProjectionSchedule.PORTABLE
    pack: int = 8
    vector_rows: int = 1

    @staticmethod
    def implementation():
        from magnitude_engine.operations.linear import EncodedLinear

        return EncodedLinear


@blueprint
class EncodedEmbedding(Blueprint[Embedding]):
    weight: Blueprint[Weight]

    @staticmethod
    def implementation():
        from magnitude_engine.operations.embedding import EncodedEmbedding

        return EncodedEmbedding


@blueprint
class DenseParameter(Blueprint[Parameter]):
    weight: Blueprint[Weight]

    @staticmethod
    def implementation():
        from magnitude_engine.operations.parameters import DenseParameter

        return DenseParameter


@blueprint
class MLXOperations(Blueprint[WeightOperations]):
    artifact: Blueprint[MLXArtifact]
    context: Blueprint[DeviceContext]

    @staticmethod
    def implementation():
        from magnitude_engine.operations.mlx import MLXOperations

        return MLXOperations
