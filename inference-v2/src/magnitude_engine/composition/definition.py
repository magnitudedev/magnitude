"""Typed construction declarations. This module has no device/runtime dependencies."""

from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import cache
from typing import (
    Any,
    Union,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import TypeAdapter


class Blueprint[T_co]:
    @staticmethod
    def implementation() -> Callable[..., T_co]:
        raise NotImplementedError("a concrete blueprint must declare its implementation")

    def describe(self) -> dict:
        from .graph import encode

        return encode(self)


@cache
def field_types(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


@cache
def result_type(cls: type[Blueprint]) -> type:
    for base in cls.__mro__:
        for declared in getattr(base, "__orig_bases__", ()):
            if get_origin(declared) is Blueprint:
                (result,) = get_args(declared)
                if isinstance(result, type):
                    return result
    raise TypeError(f"{cls.__qualname__} must declare Blueprint[LiveContract]")


def validate(value: object, annotation: Any) -> object:
    """Validate without reconstructing child nodes or losing their reference identity."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Blueprint:
        if not isinstance(value, Blueprint):
            raise TypeError("a component dependency must be a blueprint")
        (expected,) = args
        actual = result_type(type(value))
        if actual is not expected and not issubclass(actual, expected):
            raise TypeError(f"expected a blueprint for {expected.__name__}, got {actual.__name__}")
        return value
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return validate(value, option)
            except (ValueError, TypeError):
                pass
        raise TypeError(f"value does not satisfy {annotation}")
    if origin is tuple:
        if not isinstance(value, tuple):
            raise TypeError("expected a tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            for item in value:
                validate(item, args[0])
        else:
            if len(value) != len(args):
                raise TypeError("tuple length differs from its declaration")
            for item, expected in zip(value, args, strict=True):
                validate(item, expected)
        return value
    if annotation is Any:
        raise TypeError("blueprint inputs must have concrete types, not Any")
    return TypeAdapter(annotation).validate_python(value, strict=True)


@dataclass_transform(frozen_default=True, kw_only_default=True, eq_default=False)
def component[C: type](cls: C) -> C:
    """Make a frozen declaration; no registry mutation or implementation import."""
    if not issubclass(cls, Blueprint):
        raise TypeError("@component requires a Blueprint subclass")
    original = cls.__dict__.get("__post_init__")

    def checked(self: Blueprint) -> None:
        hints = field_types(type(self))
        for field in fields(self):  # type: ignore[arg-type]
            validate(getattr(self, field.name), hints[field.name])
        if original is not None:
            original(self)

    cast(Any, cls).__post_init__ = checked
    return dataclass(frozen=True, kw_only=True, eq=False)(cls)
