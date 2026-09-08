"""Separate dynamic array operands from immutable Python call structure."""

import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, cast

import mlx.core as mx


@dataclass(frozen=True)
class Tree:
    kind: str
    metadata: Any
    children: tuple["Tree", ...] = ()

    def rebuild(self, arrays: tuple[Any, ...]) -> Any:
        if self.kind == "array":
            return arrays[self.metadata]
        if self.kind == "static":
            kind, value = self.metadata
            return float.fromhex(value) if kind is float else value
        children = [child.rebuild(arrays) for child in self.children]
        if self.kind == "tuple":
            return tuple(children)
        if self.kind == "list":
            return children
        if self.kind == "dict":
            return dict(zip(self.metadata, children, strict=True))
        cls, names = self.metadata
        return cls(**dict(zip(names, children, strict=True)))


def flatten(value: Any, *, leaf_type: type = mx.array) -> tuple[Tree, tuple[Any, ...]]:
    arrays: list[Any] = []

    def visit(x) -> Tree:
        if isinstance(x, leaf_type):
            arrays.append(x)
            return Tree("array", len(arrays) - 1)
        if isinstance(x, (tuple, list)):
            return Tree("tuple" if isinstance(x, tuple) else "list", None, tuple(map(visit, x)))
        if isinstance(x, dict):
            if any(not isinstance(k, str) for k in x):
                raise TypeError("computation dictionaries require string keys")
            keys = tuple(sorted(x))
            return Tree("dict", keys, tuple(visit(x[k]) for k in keys))
        if is_dataclass(x) and not isinstance(x, type):
            if not cast(Any, x).__dataclass_params__.frozen:
                raise TypeError("computation configuration dataclasses must be frozen")
            names = tuple(f.name for f in fields(x) if f.init)
            return Tree("dataclass", (type(x), names), tuple(visit(getattr(x, n)) for n in names))
        if x is None or isinstance(x, (bool, int, float, str, mx.Dtype)):
            # A float's representation retains signed zero and makes NaN keying explicit.
            if isinstance(x, float):
                if not math.isfinite(x):
                    raise ValueError("static computation parameters must be finite")
            return Tree("static", (type(x), x.hex() if isinstance(x, float) else x))
        raise TypeError(f"unsupported computation argument: {type(x).__name__}")

    return visit(value), tuple(arrays)
