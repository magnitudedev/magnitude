from dataclasses import field

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.models.contracts import ProgramSource
from magnitude_engine.resources.io.blueprint import PositionalReader
from magnitude_engine.resources.io.reader import PositionalReader as Reader


@blueprint
class Head(Blueprint[ProgramSource]):
    artifact: Blueprint[LocalArtifact]
    target: Blueprint[ProgramSource]
    reader: Blueprint[Reader] = field(default_factory=PositionalReader)

    @staticmethod
    def implementation() -> type[ProgramSource]:
        from .loading import MTPSource

        return MTPSource
