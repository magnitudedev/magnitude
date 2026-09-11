"""Compile-time kernel arguments, distinct from execution operands and state."""

import inspect
import struct
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tilelang.jit import PrimFunc


def parameter_key(value: object) -> tuple:
    # Python equates True/1 and 0.0/-0.0. A kernel factory may distinguish them;
    # specialization identity must preserve their type and floating bit meaning.
    if isinstance(value, Enum):
        return type(value), parameter_key(value.value)
    if isinstance(value, float):
        return float, struct.pack("!d", value)
    if type(value) in (int, bool, str, bytes, type(None)):
        return type(value), value
    if isinstance(value, tuple):
        return tuple, tuple(parameter_key(item) for item in value)
    # Value objects (Capability, Precision, a Representation) are compile-time
    # arguments like any other: identity is their type and their fields.
    if is_dataclass(value) and not isinstance(value, type) and value.__dataclass_params__.frozen:
        return type(value), tuple(
            parameter_key(getattr(value, field.name)) for field in fields(value)
        )
    raise TypeError(
        "kernel specialization requires immutable scalar, enum, tuple or frozen value parameters"
    )


@cache
def signature(factory: Callable) -> inspect.Signature:
    return inspect.signature(factory)


@dataclass(frozen=True)
class Specialization:
    factory: Callable[..., "PrimFunc"]
    parameters: tuple[tuple[str, tuple], ...]

    @classmethod
    def bind[**P](cls, factory: Callable[P, "PrimFunc"], *args: P.args, **kwargs: P.kwargs):
        parameters = signature(factory).bind(*args, **kwargs)
        parameters.apply_defaults()
        return cls(
            factory,
            tuple((name, parameter_key(value)) for name, value in parameters.arguments.items()),
        )
