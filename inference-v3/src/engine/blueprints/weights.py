"""Weight containers and the residency that turns them into device weights."""

import ops
from engine.composition import Blueprint, blueprint
from engine.weights.formats.gguf import GGUFFormat
from engine.weights.formats.mlx_safetensors import MLXFormat
from engine.weights.tensor_residency import TensorWeights as Residency
from engine.weights.tensor_residency import StreamedWeights

__all__ = ["GGUF", "MLX", "Weights", "Streamed"]


@blueprint
class GGUF(Blueprint[GGUFFormat]):
    path: str

    @staticmethod
    def implementation():
        return GGUFFormat


@blueprint
class MLX(Blueprint[MLXFormat]):
    path: str

    @staticmethod
    def implementation():
        return MLXFormat


@blueprint
class Weights(Blueprint[Residency]):
    format: Blueprint[GGUFFormat] | Blueprint[MLXFormat]
    context: Blueprint[ops.DeviceRuntime]

    @staticmethod
    def implementation():
        def build(format: GGUFFormat | MLXFormat, context: ops.DeviceRuntime) -> Residency:
            return Residency(format, context)

        return build


@blueprint
class Streamed(Blueprint[StreamedWeights]):
    format: Blueprint[GGUFFormat] | Blueprint[MLXFormat]
    context: Blueprint[ops.DeviceRuntime]

    @staticmethod
    def implementation():
        def build(format: GGUFFormat | MLXFormat, context: ops.DeviceRuntime) -> StreamedWeights:
            return StreamedWeights(format, context)

        return build
