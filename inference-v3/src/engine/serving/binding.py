"""Serving's shared immutable input metadata and live execution dependencies."""

from dataclasses import dataclass

from engine.inputs.formats.gguf_tokenizer import TokenizerArtifact
from engine.service.engine import Engine


@dataclass(frozen=True)
class Components:
    engine: Engine
    tokenizer: TokenizerArtifact
