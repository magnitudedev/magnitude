"""Typed Python structures used for formula ports and trace signatures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable


def leaves(value: Any, path: tuple[str | int, ...] = ()):
    if is_dataclass(value) and not isinstance(value, type):
        # Tensor, TensorSpec, and Argument are leaves despite being dataclasses.
        from .tracing import Argument, Tensor
        from .types import TensorSpec

        if isinstance(value, (Tensor, TensorSpec, Argument)):
            yield path, value
        else:
            for field in fields(value):
                yield from leaves(getattr(value, field.name), (*path, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError("formula port keys must be strings or integers")
            yield from leaves(item, (*path, key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    else:
        yield path, value


def map_tree(value: Any, transform: Callable[[Any, tuple[str | int, ...]], Any], path=()):
    from .tracing import Argument, Tensor
    from .types import TensorSpec

    if isinstance(value, (Argument, Tensor, TensorSpec)):
        return transform(value, path)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{
            field.name: map_tree(getattr(value, field.name), transform, (*path, field.name))
            for field in fields(value) if field.init
        })
    if isinstance(value, Mapping):
        return {key: map_tree(item, transform, (*path, key)) for key, item in value.items()}
    if isinstance(value, tuple):
        mapped = tuple(map_tree(item, transform, (*path, i)) for i, item in enumerate(value))
        return type(value)(*mapped) if hasattr(value, "_fields") else mapped
    if isinstance(value, list):
        return [map_tree(item, transform, (*path, i)) for i, item in enumerate(value)]
    return transform(value, path)
