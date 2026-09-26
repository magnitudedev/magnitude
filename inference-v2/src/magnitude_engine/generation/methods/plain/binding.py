from magnitude_engine.generation.contracts import MethodFactory
from magnitude_engine.models.residency import BoundExecutor, ModelResources

from .runtime import PlainMethod


class Plain(MethodFactory):
    def bind(self, target: BoundExecutor, resources: ModelResources):
        return PlainMethod(), 0, None
