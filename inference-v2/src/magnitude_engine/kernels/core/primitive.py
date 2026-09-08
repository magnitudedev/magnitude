"""Owned numerical leaves retain their declaration through MLX graph capture."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metal import Binding
import mlx.core as mx

from .graph import Tensor
from .runtime import execution_context

markers: ContextVar[dict | None] = ContextVar("magnitude_numerical_markers", default=None)


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
        from .runtime import generated_kernel, kernel_name

        # This source deliberately cannot execute. The capture adapter replaces the
        # marker with this exact declaration; it never interprets numerical source.
        marker = f"magnitude_marker_{len(active)}"
        source = f"#error {marker}: capture marker escaped lowering"
        outputs = self.infer(inputs)
        names = tuple(f"a{i}" for i in range(len(arrays)))
        results = tuple(f"o{i}" for i in range(len(outputs)))
        kernel = generated_kernel(source, names, results)
        active["custom_kernel_" + kernel_name(source, names, results)] = self
        return tuple(
            kernel(
                inputs=list(arrays),
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[t.shape for t in outputs],
                output_dtypes=[t.dtype for t in outputs],
            )
        )


@lru_cache(maxsize=256)
def _specialize(operation: Primitive, inputs: tuple[Tensor, ...], _stream):
    operation.infer(inputs)
    return operation.lower(inputs)
