"""Typed analysis schemas over actual runtime bindings; no execution wrappers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import NoneType, UnionType
from typing import Any, get_args, get_type_hints

from magnitude_engine import components as c


@dataclass(frozen=True)
class Use:
    """One use of an actual object, with owner-supplied geometry or sharing context."""

    value: object
    context: object = None
    dependencies: Mapping[str, Use] = field(default_factory=dict)
    # Explicit adapters for upstream objects and individually declared operation ports.
    implementation: c.Implementation | None = None
    fields: Fields | None = None


@dataclass(frozen=True)
class Fields[P: c.Facts]:
    parameters: P
    operands: Mapping[str, object] = field(default_factory=dict)
    children: Mapping[str, Use] = field(default_factory=dict)
    dependencies: Mapping[str, Use] = field(default_factory=dict)
    sources: tuple[object, ...] = ()
    configuration: c.Configuration = field(default_factory=c.Configuration)


@dataclass(frozen=True)
class Schema[T, U, P: c.Facts]:
    owner: type[T]
    context: type[U] | UnionType
    read: Callable[[T, U], Fields[P]]

    def fields(self, value: T, context: U) -> Fields[P]:
        if not isinstance(context, self.context):
            name = (
                self.context.__qualname__ if isinstance(self.context, type) else str(self.context)
            )
            raise TypeError(f"{self.owner.__qualname__} requires {name} bindings")
        return self.read(value, context)


SCHEMAS: dict[type, Schema[Any, Any, Any]] = {}
ALIASES: dict[type, Callable[[Any], object]] = {}


def schema[T, U](owner: type[T], *, context: type[U] | UnionType = NoneType):
    def register[P: c.Facts](read: Callable[[T, U], Fields[P]]):
        if owner in SCHEMAS:
            raise TypeError(f"duplicate binding schema: {owner.__qualname__}")
        # The schema's typed result must agree with the production declaration.
        parameters = get_args(get_type_hints(read)["return"])[0]
        if c.component_of(owner).contract.parameters is not parameters:
            raise TypeError(f"wrong parameter schema for {owner.__qualname__}")
        SCHEMAS[owner] = Schema(owner, context, read)
        return read

    return register


def alias[T](owner: type[T], read: Callable[[T], object]) -> None:
    ALIASES[owner] = read


def resolve(use: Use) -> Use:
    from dataclasses import replace

    value = use.value
    seen = set()
    while type(value) in ALIASES:
        if id(value) in seen:
            raise ValueError("cyclic runtime binding")
        seen.add(id(value))
        value = ALIASES[type(value)](value)
    return replace(use, value=value)


def read(use: Use) -> tuple[Use, Fields, c.Implementation]:
    use = resolve(use)
    if use.fields is None:
        try:
            shape = SCHEMAS[type(use.value)]
        except KeyError as error:
            raise TypeError(
                f"no typed binding schema for {type(use.value).__qualname__}"
            ) from error
        fields = shape.fields(use.value, use.context)
    else:
        fields = use.fields
    implementation = use.implementation or c.component_of(use.value).identity(use.value)
    if not isinstance(fields.parameters, implementation.contract.parameters):
        raise TypeError("component and binding schema have different parameter contracts")
    return use, fields, implementation


def foreign[P: c.Facts](
    value: object, implementation: c.Implementation[P], fields: Fields[P]
) -> Use:
    """An explicit boundary for code we cannot decorate (e.g. an upstream projection)."""
    return Use(value, implementation=implementation, fields=fields)


def operation(value: object, parameters: c.Facts | None = None) -> Use:
    """A declared function/method port, invoked directly by production."""
    implementation = c.component_of(value).identity(value)
    return Use(value, fields=Fields(parameters or implementation.contract.parameters()))


def neural(
    *,
    operands: Mapping[str, object],
    children: Mapping[str, Use] | None = None,
    dependencies: Mapping[str, Use] | None = None,
    settings=None,
    weight_use=c.WeightUse.FULL,
    top_k=None,
    sources=(),
) -> Fields[c.NeuralParameters]:
    return Fields(
        c.NeuralParameters(settings=settings or {}, weight_use=weight_use, top_k=top_k),
        operands,
        children or {},
        dependencies or {},
        sources,
    )


def port(value: object, declaration: object, parameters: c.Facts | None = None) -> Use:
    identity = c.component_of(declaration).identity(declaration)
    return Use(
        value,
        implementation=identity,
        fields=Fields(parameters or identity.contract.parameters(), sources=(declaration,)),
    )
