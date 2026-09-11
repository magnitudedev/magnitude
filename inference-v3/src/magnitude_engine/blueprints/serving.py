from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
from magnitude_engine.service.engine import Engine
from magnitude_engine.serving.binding import Components
from magnitude_engine.weights.formats.gguf import GGUFFormat
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat

__all__ = ["ChatMetadata", "MLXChatMetadata", "ChatComponents"]


@blueprint
class ChatMetadata(Blueprint[TokenizerArtifact]):
    artifact: Blueprint[GGUFFormat]

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
    artifact: Blueprint[MLXFormat]

    @staticmethod
    def implementation():
        from magnitude_engine.inputs.formats.gguf_tokenizer import mlx_tokenizer

        return mlx_tokenizer
