"""Worker construction: validate wiring first, then build and retire one owned graph."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, cast

from .definition import Blueprint, field_types, validate
from .graph import encode


@contextmanager
def build[T](root: Blueprint[T]) -> Iterator[T]:
    encode(root)  # Reject cycles and non-data inputs before implementation imports.
    constructors: dict[int, Callable[..., Any]] = {}

    def prepare(item: object) -> None:
        if isinstance(item, tuple):
            for child in item:
                prepare(child)
        elif isinstance(item, Blueprint) and id(item) not in constructors:
            hints = field_types(type(item))
            args = {}
            for field in fields(item):  # type: ignore[arg-type]
                child = getattr(item, field.name)
                validate(child, hints[field.name])
                prepare(child)
                args[field.name] = child
            constructor = item.implementation()
            signature = inspect.signature(constructor)
            if any(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()
            ):
                raise TypeError("component construction requires an explicit typed signature")
            signature.bind(**args)
            constructors[id(item)] = constructor

    prepare(root)
    instances: dict[int, Any] = {}
    retire: list[Callable[[], None]] = []
    owned: set[int] = set()

    def construct(item: object) -> Any:
        if isinstance(item, tuple):
            return tuple(construct(child) for child in item)
        if not isinstance(item, Blueprint):
            return item
        key = id(item)
        if key not in instances:
            kwargs = {
                f.name: construct(getattr(item, f.name))
                for f in fields(item)  # type: ignore[arg-type]
            }
            instance = constructors[key](**kwargs)
            instances[key] = instance
            close = getattr(instance, "close", None)
            if close is not None and id(instance) not in owned:
                retire.append(close)
                owned.add(id(instance))
        return instances[key]

    failure: BaseException | None = None
    try:
        yield cast(T, construct(root))
    except BaseException as error:
        failure = error
        raise
    finally:
        errors = []
        for close in reversed(retire):
            try:
                close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "component construction/execution and cleanup failed",
                ([failure] if failure is not None else []) + errors,
            )
