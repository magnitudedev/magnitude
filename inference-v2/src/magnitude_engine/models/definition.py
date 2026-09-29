"""Production model identities own their default executor construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.composition.graph import digest
from magnitude_engine.models.executor.contracts import ExecutorFactory

if TYPE_CHECKING:
    from .executor.blueprint import Executor


@dataclass(frozen=True)
class ModelDefinition:
    identity: str
    factory: Callable[[Blueprint[LocalArtifact]], Executor]

    def default(self, artifact: Blueprint[LocalArtifact]) -> Default:
        return Default(definition=self.identity, artifact=artifact, executor=self.factory(artifact))


@blueprint
class Default(Blueprint[ExecutorFactory]):
    definition: str
    artifact: Blueprint[LocalArtifact]
    executor: Blueprint[ExecutorFactory]

    def __post_init__(self):
        expected = definition(self.definition).factory(self.artifact)
        if digest(self.executor) != digest(expected):
            raise ValueError(
                "a production default must use its definition's current construction; "
                "use an explicit executor for a candidate"
            )

    @staticmethod
    def implementation() -> type[ExecutorFactory]:
        from .executor.binding import DefaultExecutor

        return DefaultExecutor


def definition(identity: str) -> ModelDefinition:
    from .architectures.gemma4.definition import DEFINITION as gemma
    from .architectures.mlx_vlm.definition import DEFINITION as vlm
    from .architectures.qwen35.definition import DEFINITION as qwen

    for value in (qwen, gemma, vlm):
        if value.identity == identity:
            return value
    raise ValueError(f"unknown production model definition: {identity}")
