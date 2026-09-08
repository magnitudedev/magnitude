"""Owned numerical leaves retain their declaration through MLX graph capture."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from ._graph import identity
from .runtime import generated_kernel

if TYPE_CHECKING:
    from .metal import Binding
import mlx.core as mx

from .context import markers
from .graph import Tensor
from .runtime import execution_context


class Primitive(ABC):
    """An immutable numerical operation. Document its arithmetic contract on the class.

    Infer describes results; lower arranges execution without changing that contract.
    Instances must be hashable and must never retain dynamic operands.
    """

    @abstractmethod
    def infer(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]: ...

    @abstractmethod
    def lower(self, inputs: tuple[Tensor, ...]) -> Callable[..., tuple[mx.array, ...]]: ...

    def bindings(self, values) -> tuple["Binding", ...]:
        return ()

    def specialize(self, inputs):
        return _specialize(self, inputs, execution_context())

    def __call__(self, *arrays: mx.array) -> tuple[mx.array, ...]:
        inputs = tuple(Tensor(a.shape, a.dtype) for a in arrays)
        active = markers.get()
        if active is None:
            return self.specialize(inputs)(*arrays)

        # This source deliberately cannot execute. The capture adapter replaces the
        # marker with this exact declaration; it never interprets numerical source.
        marker = f"magnitude_marker_{len(active)}"
        source = f"#error {marker}: capture marker escaped lowering"
        outputs = self.infer(inputs)
        names = tuple(f"a{i}" for i in range(len(arrays)))
        results = tuple(f"o{i}" for i in range(len(outputs)))
        kernel = generated_kernel(source, names, results)
        result = tuple(
            kernel(
                inputs=list(arrays),
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[t.shape for t in outputs],
                output_dtypes=[t.dtype for t in outputs],
            )
        )

        active[identity(result[0])] = self
        return result


@lru_cache(maxsize=256)
def _specialize(operation: Primitive, inputs: tuple[Tensor, ...], _stream):
    operation.infer(inputs)
    return operation.lower(inputs)
