"""Host-side qualification policy selects a production definition's defaults."""

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint

from .architectures.mlx_vlm.definition import DEFINITION
from .executor.contracts import ExecutorFactory


def auto(artifact: Blueprint[LocalArtifact]) -> Blueprint[ExecutorFactory]:
    # Qualification still selects resident upstream execution. A promotion changes
    # this production choice, never a benchmark or presentation registry.
    return DEFINITION.default(artifact)
