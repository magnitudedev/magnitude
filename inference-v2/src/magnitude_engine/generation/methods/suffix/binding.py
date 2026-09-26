from dataclasses import dataclass

from magnitude_engine.generation.contracts import MethodFactory
from magnitude_engine.models.residency import BoundExecutor, ModelResources

from .runtime import SuffixMethod


@dataclass(eq=False)
class Suffix(MethodFactory):
    minimum: int
    maximum: int

    def bind(self, target: BoundExecutor, resources: ModelResources):
        return SuffixMethod(self.minimum, self.maximum), self.maximum, "suffix"
