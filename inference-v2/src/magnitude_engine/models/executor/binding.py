"""Compose a program and its compatible state through explicit model resources."""

from ..contracts import ProgramSource
from ..residency import BoundExecutor, ModelResources
from ..runtime import ModelRuntime
from ..state.contracts import StateFactory
from .contracts import ExecutorFactory


class ModelExecutor(ExecutorFactory):
    def __init__(self, *, program: ProgramSource, state: StateFactory):
        state.validate(program)
        self.program, self.state = program, state

    def load(self, resources: ModelResources) -> BoundExecutor:
        def construct() -> BoundExecutor:
            program = self.program.load(resources)
            states = self.state.create(program, resources)
            return BoundExecutor(ModelRuntime(program.program, states, resources.owner), program)

        return resources.once(self, construct)
