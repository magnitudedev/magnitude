"""Container interpretation and numerical runtime construction dependencies."""

from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.models.qwen35.description import DenseDescription
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.operations.binding import Operations
from magnitude_engine.weights.formats.gguf import GGUFFormat
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat

__all__ = ["Qwen35DenseDescription", "Qwen35MLXDescription", "Qwen35Dense"]


@blueprint
class Qwen35DenseDescription(Blueprint[DenseDescription]):
    format: Blueprint[GGUFFormat]

    @staticmethod
    def implementation():
        def build(format: GGUFFormat) -> DenseDescription:
            from magnitude_engine.models.qwen35.formats.gguf import describe

            return describe(format)

        return build


@blueprint
class Qwen35MLXDescription(Blueprint[DenseDescription]):
    format: Blueprint[MLXFormat]

    @staticmethod
    def implementation():
        def build(format: MLXFormat) -> DenseDescription:
            from magnitude_engine.models.qwen35.formats.mlx import describe

            return describe(format)

        return build


@blueprint
class Qwen35Dense(Blueprint[DenseRuntime]):
    description: Blueprint[DenseDescription]
    operations: Blueprint[Operations]
    precision: str = "native_bf16"

    @staticmethod
    def implementation():
        def build(
            description: DenseDescription, operations: Operations, precision: str
        ) -> DenseRuntime:
            from magnitude_engine.kernels.precision import preset

            return DenseRuntime(description, operations, preset(precision))

        return build
