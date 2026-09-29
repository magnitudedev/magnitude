"""Extract a formula from its trace, preserving ports, state and reference semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .formula import FormulaHandle, FormulaIndex
from .tensor.graph import Graph, ValueKind
from .tensor.types import TensorSpec


@dataclass(frozen=True, slots=True)
class BoundaryValue:
    original: int
    local: int
    spec: TensorSpec
    kind: ValueKind
    paths: tuple[tuple[str | int, ...], ...]


@dataclass(frozen=True, slots=True)
class IsolatedFormula:
    target: FormulaHandle
    graph: Graph
    inputs: tuple[BoundaryValue, ...]
    outputs: tuple[BoundaryValue, ...]

    def bind(self, values: Mapping[int, Any]) -> dict[int, Any]:
        """Map a prepared reference/fixture value table, not a second authored case."""
        missing = tuple(port.original for port in self.inputs if port.original not in values)
        if missing:
            raise KeyError(f"formula fixture is missing traced input values {missing}")
        return {port.local: values[port.original] for port in self.inputs}


def isolate(target: FormulaHandle) -> IsolatedFormula:
    if not isinstance(target, FormulaHandle):
        raise TypeError("isolation requires a typed formula occurrence")
    graph, call = target.graph, target.call
    if not call.complete:
        raise ValueError("isolate from the original formula trace, not a pruned partial occurrence")
    boundary = tuple(dict.fromkeys(port.value for port in call.inputs))
    available = set(boundary)
    nodes = tuple(graph.node(index) for index in call.nodes)
    for node in nodes:
        if any(value not in available for value in node.inputs):
            raise ValueError("formula has an undeclared dependency outside its boundary")
        available.update(node.outputs)
    if any(port.value not in available for port in call.outputs):
        raise ValueError("formula output is outside its declared computation")

    ordered_values = (*boundary, *(value for node in nodes for value in node.outputs))
    value_ids = {value: index for index, value in enumerate(ordered_values)}
    node_ids = {node.id: index for index, node in enumerate(nodes)}
    boundary_set = frozenset(boundary)
    values = []
    for original in ordered_values:
        value = graph.value(original)
        if original in boundary_set:
            kind = (ValueKind.RESOURCE if value.resource_id is not None else
                    ValueKind.CONSTANT if value.kind == ValueKind.CONSTANT else ValueKind.INPUT)
            values.append(replace(value, id=value_ids[original], kind=kind,
                                  producer=None, output_index=0))
        else:
            values.append(replace(value, id=value_ids[original], producer=node_ids[value.producer]))
    mapped_nodes = tuple(replace(
        node, id=node_ids[node.id], inputs=tuple(value_ids[value] for value in node.inputs),
        outputs=tuple(value_ids[value] for value in node.outputs),
    ) for node in nodes)
    descendant_ids = {call.occurrence}
    for child in graph.formulas:
        if child.parent in descendant_ids:
            descendant_ids.add(child.occurrence)
    calls = tuple(replace(
        child.remap(value_ids, node_ids), parent=None if child is call else child.parent,
    ) for child in graph.formulas if child.occurrence in descendant_ids)
    isolated = Graph(
        call.formula.id, tuple(values), mapped_nodes,
        tuple(value.id for value in values if value.kind == ValueKind.INPUT),
        tuple(value.id for value in values if value.kind == ValueKind.CONSTANT),
        tuple(value.id for value in values if value.kind == ValueKind.RESOURCE),
        tuple(value_ids[port.value] for port in call.outputs), FormulaIndex(calls),
    )

    def ports(items):
        return tuple(BoundaryValue(
            original, value_ids[original], graph.value(original).spec,
            isolated.value(value_ids[original]).kind,
            tuple(port.path for port in items if port.value == original),
        ) for original in dict.fromkeys(port.value for port in items))

    return IsolatedFormula(target, isolated, ports(call.inputs), ports(call.outputs))
