"""Default owned Gemma composition."""

from magnitude_engine.models.definition import ModelDefinition


def default(artifact):
    from magnitude_engine.models.executor.blueprint import Executor
    from magnitude_engine.models.state.blueprint import PagedHybrid

    from .blueprint import Program

    return Executor(program=Program(artifact=artifact), state=PagedHybrid())


DEFINITION = ModelDefinition("GEMMA4", default)
