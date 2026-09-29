"""The qualified automatic default and generic upstream model identity."""

from magnitude_engine.models.definition import ModelDefinition


def default(artifact):
    from magnitude_engine.models.executor.blueprint import Executor
    from magnitude_engine.models.state.blueprint import Native

    from .blueprint import Forward, ModelLoader

    source = ModelLoader(artifact=artifact)
    return Executor(program=Forward(source=source), state=Native(source=source))


DEFINITION = ModelDefinition("MLX_VLM", default)
