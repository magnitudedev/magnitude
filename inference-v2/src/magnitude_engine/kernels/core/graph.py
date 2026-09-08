"""Immutable numerical dataflow, independent of Metal execution arrangement."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
    from .primitive import Primitive


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    dtype: mx.Dtype

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in self.shape):
            raise ValueError("tensor shapes require nonnegative integer extents")

    @property
    def size(self) -> int:
        from math import prod

        return prod(self.shape)


@dataclass(frozen=True)
class Value:
    name: str
    tensor: Tensor


@dataclass(frozen=True)
class Node:
    operation: "str | Primitive"
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    attributes: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Graph:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    constants: tuple[tuple[Value, mx.array], ...]
    nodes: tuple[Node, ...]

    def __post_init__(self):
        defined = {v.name: v for v in self.inputs}
        for value, _ in self.constants:
            if value.name in defined:
                raise ValueError(f"duplicate graph value: {value.name}")
            defined[value.name] = value
        for node in self.nodes:
            if not node.outputs:
                raise ValueError("a numerical node must produce a value")
            for value in node.inputs:
                if defined.get(value.name) != value:
                    raise ValueError(f"undefined or inconsistent input: {value.name}")
            for value in node.outputs:
                if value.name in defined:
                    raise ValueError(f"duplicate graph value: {value.name}")
                defined[value.name] = value
        for value in self.outputs:
            if defined.get(value.name) != value:
                raise ValueError(f"undefined graph output: {value.name}")

    def describe(self) -> str:
        return "\n".join(
            f"{', '.join(v.name for v in n.outputs)} = {n.operation}"
            f"({', '.join(v.name for v in n.inputs)}) {n.attributes!r}"
            for n in self.nodes
        )


def signature(names: tuple[str, ...], tensors: tuple[Tensor, ...]) -> tuple[Value, ...]:
    return tuple(Value(name, tensor) for name, tensor in zip(names, tensors, strict=True))
