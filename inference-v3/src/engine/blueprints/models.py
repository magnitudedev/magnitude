"""Container interpretation and numerical runtime construction dependencies."""

import ops
from engine.composition import Blueprint, blueprint
from engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
from engine.loading import LoadedModel
from engine.models.qwen35.description import DenseDescription
from engine.models.qwen35.runtime import DenseRuntime
from engine.models.sequence import ModelExecutor
from engine.weights.formats.gguf import GGUFFormat
from engine.weights.formats.mlx_safetensors import MLXFormat
from engine.weights.residency import WeightResidency

__all__ = ["Qwen35DenseDescription", "Qwen35MLXDescription", "Qwen35Dense", "LoadedComponents"]


@blueprint
class Qwen35DenseDescription(Blueprint[DenseDescription]):
    format: Blueprint[GGUFFormat]

    @staticmethod
    def implementation():
        def build(format: GGUFFormat) -> DenseDescription:
            from engine.models.qwen35.formats.gguf import describe

            return describe(format)

        return build


@blueprint
class Qwen35MLXDescription(Blueprint[DenseDescription]):
    format: Blueprint[MLXFormat]

    @staticmethod
    def implementation():
        def build(format: MLXFormat) -> DenseDescription:
            from engine.models.qwen35.formats.mlx import describe

            return describe(format)

        return build


@blueprint
class Qwen35Dense(Blueprint[DenseRuntime]):
    description: Blueprint[DenseDescription]
    device: Blueprint[ops.DeviceRuntime]
    weights: Blueprint[WeightResidency]
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


@blueprint
class LoadedComponents(Blueprint[LoadedModel]):
    executor: Blueprint[ModelExecutor]
    metadata: Blueprint[TokenizerArtifact]

    @staticmethod
    def implementation():
        from engine.inputs.tokenizer import ByteBPETokenizer

        def construct(executor, metadata) -> LoadedModel:
            return LoadedModel(executor, ByteBPETokenizer(metadata.config))

        return construct
