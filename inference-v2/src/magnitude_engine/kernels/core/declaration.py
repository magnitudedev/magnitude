"""The single kernel authoring API: validated tensor bindings to handwritten Metal."""

from dataclasses import dataclass
from functools import lru_cache, update_wrapper
from pathlib import Path
from typing import Any

from .execution import OperandBinding
from .graph import Graph, Node, Tensor, signature
from .kernel import Kernel as OpaqueKernel
from .metal import Binding, Dispatch, TensorSpec, TileCall
from .plan import Source
from .primitive import Primitive
from .scheduling import Automatic
from .trees import Tree, flatten


class Kernel:
    def __init__(self, function, source, entry):
        self.function = function
        if isinstance(source, Source):
            self.source = source
        else:
            path = (Path(function.__code__.co_filename).resolve().parent / source).resolve()
            package = Path(__file__).resolve().parent.parent
            self.source = Source(
                str(path.relative_to(package)) if path.is_relative_to(package) else str(path)
            )
        self.entry = entry
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs) -> Any:
        tree, arrays = flatten((args, kwargs))
        call = Call(self, tree)
        result = call(*arrays)
        return result[0] if len(result) == 1 else result


@dataclass(frozen=True)
class Call(Primitive):
    declaration: Kernel
    tree: Tree

    def describe(self, values):
        args, kwargs = self.tree.rebuild(tuple(TensorSpec(v) for v in values))
        return self.declaration.function(*args, **kwargs)

    def infer(self, inputs):
        description = describe(self, inputs)
        if isinstance(description, Dispatch):
            return tuple(description.outputs.values())
        if isinstance(description, TileCall):
            return (Tensor(description.domain.shape, description.result.dtype),)
        return (description.output,)

    def bindings(self, values):
        description = self.describe(values)
        if isinstance(description, Dispatch):
            return ()
        if self.declaration.entry is None:
            raise ValueError("a composable Metal function requires its entry name")
        return (Binding(self.declaration.source, self.declaration.entry, description),)

    def lower(self, inputs):
        description = describe(self, inputs)
        if isinstance(description, Dispatch):
            values = tuple(v.value for v in description.arguments.values())
            # The named Metal ABI is independent of the Python operand ordering.
            named = signature(tuple(description.arguments), tuple(v.tensor for v in values))
            kernel = OpaqueKernel(
                named,
                signature(tuple(description.outputs), tuple(description.outputs.values())),
                self.declaration.source,
                description.launch,
                description.template,
                description.constants,
            ).bind()
            operands = tuple(int(v.name[1:]) for v in values)
            return OperandBinding(kernel, operands, tuple(range(len(description.outputs))))
        values = signature(tuple(f"a{i}" for i in range(len(inputs))), inputs)
        outputs = signature(
            tuple(f"o{i}" for i in range(len(self.infer(inputs)))), self.infer(inputs)
        )
        return Automatic().lower(Graph(values, outputs, (), (Node(self, values, outputs),)))


@lru_cache(maxsize=256)
def describe(call, tensors):
    return call.describe(signature(tuple(f"a{i}" for i in range(len(tensors))), tensors))


def kernel(*, source, function=None):
    return lambda definition: Kernel(definition, source, function)
