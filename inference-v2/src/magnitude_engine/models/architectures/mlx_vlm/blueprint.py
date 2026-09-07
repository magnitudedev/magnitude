from dataclasses import field

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.models.contracts import ProgramSource
from magnitude_engine.resources.io.blueprint import PositionalReader
from magnitude_engine.resources.io.reader import PositionalReader as Reader


@blueprint
class ModelLoader(Blueprint[ProgramSource]):
    artifact: Blueprint[LocalArtifact]
    reader: Blueprint[Reader] = field(default_factory=PositionalReader)

    @staticmethod
    def implementation() -> type[ProgramSource]:
        from .loading import UpstreamLoader

        return UpstreamLoader


@blueprint
class Forward(Blueprint[ProgramSource]):
    source: Blueprint[ProgramSource]

    @staticmethod
    def implementation() -> type[ProgramSource]:
        from .loading import UpstreamForward

        return UpstreamForward
