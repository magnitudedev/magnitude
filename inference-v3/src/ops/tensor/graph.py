"""Immutable first-order tensor graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .types import Dim, Layout, TensorSpec
from ..formula import FormulaIndex


class ValueKind(StrEnum):
    INPUT = "input"
    CONSTANT = "constant"
    RESOURCE = "resource"
    NODE = "node"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    filename: str
    line: int
    function: str


@dataclass(frozen=True, slots=True)
class Value:
    id: int
    spec: TensorSpec
    kind: ValueKind
    name: str | None = None
    producer: int | None = None
    output_index: int = 0
    resource_id: int | None = None
    resource_version: int | None = None


@dataclass(frozen=True, slots=True)
class Effects:
    reads: tuple[int, ...] = ()
    writes: tuple[tuple[int, int, int], ...] = ()
    aliases: tuple[tuple[int, int], ...] = ()
    host_observation: bool = False


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    operation: str
    inputs: tuple[int, ...]
    attributes: Mapping[str, Any]
    outputs: tuple[int, ...]
    effects: Effects = Effects()
    source: SourceLocation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class Graph:
    name: str
    values: tuple[Value, ...]
    nodes: tuple[Node, ...]
    inputs: tuple[int, ...]
    constants: tuple[int, ...]
    resources: tuple[int, ...]
    outputs: tuple[int, ...]
    formulas: FormulaIndex = FormulaIndex()
    _users: tuple[tuple[int, ...], ...] = field(init=False, repr=False, compare=False)
    _fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if tuple(value.id for value in self.values) != tuple(range(len(self.values))):
            raise ValueError("graph value identifiers must be dense and ordered")
        if tuple(node.id for node in self.nodes) != tuple(range(len(self.nodes))):
            raise ValueError("graph node identifiers must be dense and ordered")
        available = set((*self.inputs, *self.constants, *self.resources))
        for node in self.nodes:
            if any(value not in available for value in node.inputs):
                raise ValueError(f"node {node.id} reads a value before it is defined")
            available.update(node.outputs)
        if any(value not in available for value in self.outputs):
            raise ValueError("graph output is not defined")
        users: list[list[int]] = [[] for _ in self.values]
        for node in self.nodes:
            for value in node.inputs:
                users[value].append(node.id)
        object.__setattr__(self, "_users", tuple(tuple(items) for items in users))

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is not None:
            return self._fingerprint
        encoded = json.dumps(_stable(self), sort_keys=True, separators=(",", ":")).encode()
        result = hashlib.sha256(encoded).hexdigest()
        object.__setattr__(self, "_fingerprint", result)
        return result

    @property
    def users(self) -> tuple[tuple[int, ...], ...]:
        return self._users

    def value(self, value_id: int) -> Value:
        return self.values[value_id]

    def node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def alias_root(self, identity: int) -> int:
        """Backing identity guaranteed by primitive effects, not chosen by a kernel."""
        while self.value(identity).producer is not None:
            value = self.value(identity)
            node = self.node(value.producer)
            source = dict(node.effects.aliases).get(value.output_index)
            if source is None:
                break
            identity = node.inputs[source]
        return identity


def prune_dead_nodes(graph: Graph) -> Graph:
    """Remove pure computations that cannot affect an output or observable resource."""
    live_values = set(graph.outputs)
    live_nodes: set[int] = set()
    for node in reversed(graph.nodes):
        observable = bool(node.effects.writes) or node.effects.host_observation
        if observable or any(output in live_values for output in node.outputs):
            live_nodes.add(node.id)
            live_values.update(node.outputs)
            live_values.update(node.inputs)

    declared = set((*graph.inputs, *graph.constants, *graph.resources))
    retained_values = declared | live_values
    value_ids = {
        value.id: index
        for index, value in enumerate(
            value for value in graph.values if value.id in retained_values
        )
    }
    node_ids = {
        node.id: index
        for index, node in enumerate(node for node in graph.nodes if node.id in live_nodes)
    }
    values = tuple(
        replace(
            value,
            id=value_ids[value.id],
            producer=None if value.producer is None else node_ids[value.producer],
        )
        for value in graph.values
        if value.id in retained_values
    )
    nodes = tuple(
        replace(
            node,
            id=node_ids[node.id],
            inputs=tuple(value_ids[value] for value in node.inputs),
            outputs=tuple(value_ids[value] for value in node.outputs),
        )
        for node in graph.nodes
        if node.id in live_nodes
    )
    return Graph(
        graph.name,
        values,
        nodes,
        tuple(value_ids[value] for value in graph.inputs),
        tuple(value_ids[value] for value in graph.constants),
        tuple(value_ids[value] for value in graph.resources),
        tuple(value_ids[value] for value in graph.outputs),
        FormulaIndex(tuple(call.remap(value_ids, node_ids) for call in graph.formulas)),
    )


def _stable(value: Any) -> Any:
    if isinstance(value, Graph):
        return {
            "name": value.name,
            "values": [_stable(item) for item in value.values],
            "nodes": [_stable(item) for item in value.nodes],
            "inputs": value.inputs,
            "constants": value.constants,
            "resources": value.resources,
            "outputs": value.outputs,
            "formulas": _stable(value.formulas),
        }
    if isinstance(value, Value):
        return {
            "id": value.id,
            "spec": _stable(value.spec),
            "kind": value.kind.value,
            "name": value.name,
            "producer": value.producer,
            "output_index": value.output_index,
            "resource_id": value.resource_id,
            "resource_version": value.resource_version,
        }
    if isinstance(value, Node):
        return {
            "id": value.id,
            "operation": value.operation,
            "inputs": value.inputs,
            "attributes": _stable(dict(value.attributes)),
            "outputs": value.outputs,
            "effects": _stable(value.effects),
        }
    if isinstance(value, Effects):
        return {
            "reads": value.reads,
            "writes": value.writes,
            "aliases": value.aliases,
            "host_observation": value.host_observation,
        }
    if isinstance(value, TensorSpec):
        return {
            "shape": [_stable(item) for item in value.shape],
            "dtype": value.dtype.value,
            "layout": _stable(value.layout),
            "representation": _stable(value.representation),
        }
    if isinstance(value, Dim):
        return {"dim": value.name, "minimum": value.minimum, "maximum": value.maximum}
    if isinstance(value, Layout):
        return {"strides": value.strides, "tag": value.tag}
    if hasattr(value, "__dataclass_fields__"):
        return {name: _stable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"{type(value).__qualname__} has no stable mathematical encoding")
