from magnitude_engine.composition import Blueprint, blueprint

from ..contracts import ProgramSource
from ..state.contracts import StateFactory
from .contracts import ExecutorFactory


@blueprint
class Executor(Blueprint[ExecutorFactory]):
    program: Blueprint[ProgramSource]
    state: Blueprint[StateFactory]

    @staticmethod
    def implementation() -> type[ExecutorFactory]:
        from .binding import ModelExecutor

        return ModelExecutor
