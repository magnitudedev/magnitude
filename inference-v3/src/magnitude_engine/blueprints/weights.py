"""Weight containers and the residency that turns them into device weights."""

from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.platform.execution import DeviceContext
from magnitude_engine.weights.formats.gguf import GGUFFormat
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat
from magnitude_engine.weights.residency import Weights as Residency

__all__ = ["GGUF", "MLX", "Weights"]


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
    context: Blueprint[DeviceContext]

    @staticmethod
    def implementation():
        return Residency
