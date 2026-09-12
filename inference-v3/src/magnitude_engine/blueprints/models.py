"""Container interpretation and numerical runtime construction dependencies."""

import magnitensor as mt
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.models.qwen35.description import DenseDescription
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.weights.formats.gguf import GGUFFormat
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat
from magnitude_engine.weights.tensor_residency import TensorWeights

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
    device: Blueprint[mt.Device]
    weights: Blueprint[TensorWeights]
    max_sequences: int = 8
    prefill_rows: int | None = None
    context_capacity: int | None = None

    @staticmethod
    def implementation():
        def build(
            description,
            device,
            weights,
            max_sequences,
            prefill_rows,
            context_capacity,
        ) -> DenseRuntime:
            return DenseRuntime(
                description,
                device,
                weights,
                max_sequences=max_sequences,
                prefill_rows=prefill_rows,
                context_capacity=context_capacity,
            )

        return build
