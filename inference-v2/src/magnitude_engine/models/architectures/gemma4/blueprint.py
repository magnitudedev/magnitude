from dataclasses import field

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.models.attention.blueprint import Paged
from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.contracts import ProgramSource
from magnitude_engine.models.embeddings.blueprint import Resident as ResidentEmbedding
from magnitude_engine.models.embeddings.contracts import EmbeddingFactory
from magnitude_engine.models.experts.blueprint import Resident as ResidentExperts
from magnitude_engine.models.experts.contracts import ExpertFactory
from magnitude_engine.resources.io.blueprint import PositionalReader
from magnitude_engine.resources.io.reader import PositionalReader as Reader


@blueprint
class Program(Blueprint[ProgramSource]):
    artifact: Blueprint[LocalArtifact]
    attention: Blueprint[PagedAttention] = field(default_factory=Paged)
    embedding: Blueprint[EmbeddingFactory] = field(default_factory=ResidentEmbedding)
    per_layer_embedding: Blueprint[EmbeddingFactory] = field(default_factory=ResidentEmbedding)
    experts: Blueprint[ExpertFactory] = field(default_factory=ResidentExperts)
    reader: Blueprint[Reader] = field(default_factory=PositionalReader)

    @staticmethod
    def implementation() -> type[ProgramSource]:
        from .loading import Gemma4Source

        return Gemma4Source
