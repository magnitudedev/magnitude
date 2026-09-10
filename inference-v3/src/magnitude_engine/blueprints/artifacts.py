from magnitude_engine.artifacts.mlx import MLXArtifact
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.composition import Blueprint, blueprint

__all__ = ["GGUF", "MLX"]


@blueprint
class GGUF(Blueprint[GGUFArtifact]):
    path: str

    @staticmethod
    def implementation():
        return GGUFArtifact


@blueprint
class MLX(Blueprint[MLXArtifact]):
    path: str

    @staticmethod
    def implementation():
        return MLXArtifact
