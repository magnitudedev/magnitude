from dataclasses import field

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.models.contracts import ProgramSource
from magnitude_engine.models.embeddings.blueprint import Resident
from magnitude_engine.models.embeddings.contracts import EmbeddingFactory
from magnitude_engine.resources.io.blueprint import PositionalReader
from magnitude_engine.resources.io.reader import PositionalReader as Reader

from .attention.blueprint import Attention
from .contracts import (
    AttentionFactory,
    FeedForwardFactory,
    RecurrentFactory,
)
from .feedforward.blueprint import MoE
from .recurrence.blueprint import Mixer


@component
class Program(Blueprint[ProgramSource]):
    artifact: Blueprint[LocalArtifact]
    embedding: Blueprint[EmbeddingFactory] = field(default_factory=Resident)
    attention: Blueprint[AttentionFactory] = field(default_factory=Attention)
    recurrence: Blueprint[RecurrentFactory] = field(default_factory=Mixer)
    feedforward: Blueprint[FeedForwardFactory] = field(default_factory=MoE)
    reader: Blueprint[Reader] = field(default_factory=PositionalReader)

    @staticmethod
    def implementation() -> type[ProgramSource]:
        from .loading import Qwen35Source

        return Qwen35Source
