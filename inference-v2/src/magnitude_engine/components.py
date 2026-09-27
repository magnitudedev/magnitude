"""Canonical identities declared on executable classes and operations."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from magnitude_engine.models.definition import ModelDefinition


class ComponentId(str):
    """Validated portable identity; hierarchy edges come from runtime bindings."""

    def __new__(cls, value: str):
        if not re.fullmatch(
            r"[A-Z][A-Z0-9_]*:[A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)*:(?:MLX|LM|VLM|MAG):[A-Z][A-Z0-9_]*",
            value,
        ):
            raise ValueError(f"invalid component ID: {value}")
        return super().__new__(cls, value)

    @property
    def kind(self) -> str:
        return self.rsplit(":", 2)[0]


@dataclass(frozen=True)
class Component:
    id: ComponentId
    model: ModelDefinition | None = None


# Collision detection is derived from declarations, never a second catalog.
_owners: dict[ComponentId, object] = {}


def component[T](identity: str | type, *, model: ModelDefinition | None = None) -> Callable[[T], T]:
    """Declare once, or explicitly share an existing class's identity. Never wraps execution."""
    declaration = (
        Component(ComponentId(identity), model)
        if isinstance(identity, str)
        else component_of(identity)
    )

    def declare(owner: T) -> T:
        if "__component__" in vars(owner):
            raise TypeError("component already declared")
        if isinstance(identity, str):
            previous = _owners.get(declaration.id)
            if previous is not None and previous is not owner:
                raise ValueError(f"component ID already declared: {identity}")
            _owners[declaration.id] = owner
        cast(Any, owner).__component__ = declaration
        return owner

    return declare


def component_of(value: object) -> Component:
    owner = value.__func__ if inspect.ismethod(value) else value
    namespace = (
        vars(owner) if inspect.isfunction(owner) or isinstance(owner, type) else vars(type(owner))
    )
    declaration = namespace.get("__component__")
    if not isinstance(declaration, Component):
        raise TypeError(f"undeclared execution component: {value!r}")
    return declaration


def component_id(value: object) -> ComponentId:
    return component_of(value).id
