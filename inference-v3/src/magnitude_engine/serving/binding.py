"""Serving's shared immutable input metadata and live execution dependencies."""

from dataclasses import dataclass

from magnitude_engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
from magnitude_engine.service.engine import Engine


@dataclass(frozen=True)
class Components:
    engine: Engine
    tokenizer: TokenizerArtifact
