"""Artifact interpretation and numerical runtime construction dependencies."""

from magnitude_engine.artifacts.mlx import MLXArtifact
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.models.qwen35.artifact import DenseArtifact
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.numerics.policy import NumericalFamily
from magnitude_engine.operations.factory import WeightOperations

__all__ = ["Qwen35DenseDescription", "Qwen35MLXDescription", "Qwen35Dense"]


def describe_qwen35(artifact: GGUFArtifact) -> DenseArtifact:
    from magnitude_engine.models.qwen35.artifact import inspect_dense

    return inspect_dense(artifact.directory, artifact.identity)


@blueprint
class Qwen35DenseDescription(Blueprint[DenseArtifact]):
    artifact: Blueprint[GGUFArtifact]

    @staticmethod
    def implementation():
        return describe_qwen35


@blueprint
class Qwen35Dense(Blueprint[DenseRuntime]):
    description: Blueprint[DenseArtifact]
    operations: Blueprint[WeightOperations]
    numerics: NumericalFamily = NumericalFamily.NATIVE_BF16

    @staticmethod
    def implementation():
        return DenseRuntime


@blueprint
class Qwen35MLXDescription(Blueprint[DenseArtifact]):
    artifact: Blueprint[MLXArtifact]

    @staticmethod
    def implementation():
        from magnitude_engine.models.qwen35.mlx import inspect_mlx

        return inspect_mlx
