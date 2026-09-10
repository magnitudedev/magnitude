from magnitude_engine.artifacts.mlx import MLXArtifact
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.service.engine import Engine
from magnitude_engine.serving.binding import Components

__all__ = ["ChatMetadata", "MLXChatMetadata", "ChatComponents"]


@blueprint
class ChatMetadata(Blueprint[TokenizerArtifact]):
    artifact: Blueprint[GGUFArtifact]

    @staticmethod
    def implementation():
        return TokenizerArtifact.interpret


@blueprint
class ChatComponents(Blueprint[Components]):
    engine: Blueprint[Engine]
    tokenizer: Blueprint[TokenizerArtifact]

    @staticmethod
    def implementation():
        return Components


@blueprint
class MLXChatMetadata(Blueprint[TokenizerArtifact]):
    artifact: Blueprint[MLXArtifact]

    @staticmethod
    def implementation():
        from magnitude_engine.artifacts.tokenizer import mlx_tokenizer

        return mlx_tokenizer
