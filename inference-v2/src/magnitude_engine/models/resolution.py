"""Host-side opinions produce ordinary, inspectable concrete composition."""

from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint

from .architectures.mlx_vlm.blueprint import Forward, ModelLoader
from .executor.blueprint import Executor
from .state.blueprint import Native


def auto(artifact: Blueprint[LocalArtifact]) -> Executor:
    # The qualification gate currently selects resident upstream execution.
    # Architecture/cache compatibility is finalized by the upstream binding.
    source = ModelLoader(artifact=artifact)
    return Executor(program=Forward(source=source), state=Native(source=source))
