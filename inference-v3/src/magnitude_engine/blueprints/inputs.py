"""Typed text preparation dependencies bound from an immutable artifact."""

from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.inputs.tokenizer import BPEConfig, Tokenizer
from magnitude_engine.weights.formats.gguf import GGUFFormat

__all__ = ["Qwen35Tokenization", "ByteBPE"]


@blueprint
class Qwen35Tokenization(Blueprint[BPEConfig]):
    artifact: Blueprint[GGUFFormat]

    @staticmethod
    def implementation():
        from magnitude_engine.inputs.formats.gguf_tokenizer import qwen35_tokenizer

        return qwen35_tokenizer


@blueprint
class ByteBPE(Blueprint[Tokenizer]):
    config: Blueprint[BPEConfig]

    @staticmethod
    def implementation():
        from magnitude_engine.inputs.tokenizer import ByteBPETokenizer

        return ByteBPETokenizer
