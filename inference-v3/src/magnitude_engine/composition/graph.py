"""An identity-preserving data graph, resolved through explicit public exports."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import fields
from enum import Enum
from types import ModuleType, UnionType
from typing import Union, get_args, get_origin

from .definition import Blueprint, field_types

MAX_NODES = 8192
MAX_DEPTH = 128
MAX_BYTES = 4 << 20


def identity(cls: type[Blueprint]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


class Catalog:
    def __init__(self, declarations: Iterable[type[Blueprint]]):
        self.types: dict[str, type[Blueprint]] = {}
        for declaration in declarations:
            name = identity(declaration)
            previous = self.types.setdefault(name, declaration)
            if previous is not declaration:
                raise ValueError(f"duplicate blueprint identity: {name}")

    @classmethod
    def exports(cls, namespace: ModuleType) -> Catalog:
        declarations: set[type[Blueprint]] = set()
        visited: set[ModuleType] = set()

        def collect(module: ModuleType) -> None:
            if module in visited:
                return
            visited.add(module)
            for name in module.__all__:
                item = getattr(module, name)
                if isinstance(item, ModuleType):
                    collect(item)
                elif isinstance(item, type) and issubclass(item, Blueprint):
                    declarations.add(item)

        collect(namespace)
        return cls(declarations)


def encode(root: Blueprint) -> dict:
    records: list[dict] = []
    identifiers: dict[int, str] = {}
    visiting: set[int] = set()

    def value(item: object, depth: int) -> object:
        if depth > MAX_DEPTH:
            raise ValueError("blueprint graph exceeds maximum depth")
        if isinstance(item, Blueprint):
            key = id(item)
            if key in visiting:
                raise ValueError("blueprint graph contains a cycle")
            if key not in identifiers:
                if len(records) >= MAX_NODES:
                    raise ValueError("blueprint graph exceeds maximum node count")
                name = identifiers[key] = f"n{len(records)}"
                record = {"id": name, "type": identity(type(item)), "fields": {}}
                records.append(record)
                visiting.add(key)
                record["fields"] = {
                    f.name: value(getattr(item, f.name), depth + 1)
                    for f in fields(item)  # type: ignore[arg-type]
                }
                visiting.remove(key)
            return {"ref": identifiers[key]}
        if isinstance(item, tuple):
            return {"tuple": [value(child, depth + 1) for child in item]}
        if isinstance(item, Enum):
            return {
                "enum": f"{type(item).__module__}.{type(item).__qualname__}",
                "value": value(item.value, depth + 1),
            }
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        raise TypeError(f"unsupported blueprint value: {type(item).__name__}")

    encoded_root = value(root, 0)
    if not isinstance(encoded_root, dict) or "ref" not in encoded_root:
        raise TypeError("graph root must be a blueprint")
    return {"root": encoded_root["ref"], "nodes": records}


def dumps(root: Blueprint) -> str:
    payload = json.dumps(encode(root), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode()) > MAX_BYTES:
        raise ValueError("blueprint payload exceeds maximum size")
    return payload


def digest(root: Blueprint) -> str:
    return hashlib.sha256(dumps(root).encode()).hexdigest()


def loads(payload: str, catalog: Catalog | None = None) -> Blueprint:
    if len(payload.encode()) > MAX_BYTES:
        raise ValueError("blueprint payload exceeds maximum size")
    if catalog is None:
        from magnitude_engine import blueprints

        catalog = Catalog.exports(blueprints)

    def unique(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    raw = json.loads(payload, object_pairs_hook=unique)
    if not isinstance(raw, dict) or set(raw) != {"root", "nodes"}:
        raise ValueError("invalid blueprint graph envelope")
    if not isinstance(raw["nodes"], list) or not 0 < len(raw["nodes"]) <= MAX_NODES:
        raise ValueError("invalid blueprint node count")
    records: dict[str, dict] = {}
    for record in raw["nodes"]:
        if not isinstance(record, dict) or set(record) != {"id", "type", "fields"}:
            raise ValueError("invalid blueprint node")
        name, kind = record["id"], record["type"]
        if not isinstance(name, str) or name in records:
            raise ValueError("duplicate or invalid blueprint node identity")
        if not isinstance(kind, str) or kind not in catalog.types:
            raise ValueError("unknown blueprint type")
        declaration = catalog.types[kind]
        data = record["fields"]
        expected_fields = {f.name for f in fields(declaration)}  # type: ignore[arg-type]
        if not isinstance(data, dict) or set(data) != expected_fields:
            raise ValueError("blueprint fields differ from declaration")
        records[name] = record

    visiting: set[str] = set()
    built: dict[str, Blueprint] = {}

    def value(item: object, depth: int, annotation: object = None) -> object:
        if depth > MAX_DEPTH:
            raise ValueError("blueprint graph exceeds maximum depth")
        if isinstance(item, dict):
            if set(item) == {"enum", "value"}:
                candidates = (
                    get_args(annotation)
                    if get_origin(annotation) in (Union, UnionType)
                    else (annotation,)
                )
                for candidate in candidates:
                    if (
                        isinstance(candidate, type)
                        and issubclass(candidate, Enum)
                        and item["enum"] == f"{candidate.__module__}.{candidate.__qualname__}"
                    ):
                        return candidate(item["value"])
                raise ValueError("enum value requires its declared enum type")
            if set(item) == {"ref"}:
                return node(item["ref"], depth + 1)
            if set(item) == {"tuple"} and isinstance(item["tuple"], list):
                arguments = get_args(annotation)
                if get_origin(annotation) is not tuple:
                    raise ValueError("tuple value requires a declared tuple field")
                if len(arguments) == 2 and arguments[1] is Ellipsis:
                    types = (arguments[0],) * len(item["tuple"])
                else:
                    types = arguments
                if len(types) != len(item["tuple"]):
                    raise ValueError("tuple length differs from its declaration")
                return tuple(
                    value(child, depth + 1, expected)
                    for child, expected in zip(item["tuple"], types, strict=True)
                )
            raise ValueError("invalid blueprint field encoding")
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        raise ValueError("invalid blueprint scalar")

    def node(name: object, depth: int) -> Blueprint:
        if not isinstance(name, str) or name not in records:
            raise ValueError("dangling blueprint reference")
        if name in visiting:
            raise ValueError("blueprint graph contains a cycle")
        if name not in built:
            visiting.add(name)
            record = records[name]
            declaration = catalog.types[record["type"]]
            hints = field_types(declaration)
            kwargs = {key: value(item, depth, hints[key]) for key, item in record["fields"].items()}
            built[name] = declaration(**kwargs)
            visiting.remove(name)
        return built[name]

    root = node(raw["root"], 0)
    if len(built) != len(records):
        raise ValueError("unreachable blueprint nodes")
    return root
