"""Stable mathematical definitions and occurrences in the existing tensor graph.

Formula bodies use ordinary Python and tensor primitives. They never choose or
execute a physical implementation. The expanded graph is also the sole source
for references, useful-work accounting and implementation matching.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import update_wrapper
from typing import TYPE_CHECKING, Any

from .tensor.tree import leaves
from .tensor.types import ShapeDim

if TYPE_CHECKING:
    from .tensor.graph import Graph


@dataclass(frozen=True, slots=True)
class Unit:
    name: str
    dimension: str


class units:
    token = Unit("token", "count")
    element = Unit("element", "count")
    flop = Unit("FLOP", "floating-work")
    integer_op = Unit("integer-op", "integer-work")
    special_op = Unit("special-function", "special-function-work")
    comparison = Unit("comparison", "comparison-work")
    byte = Unit("byte", "storage")
    second = Unit("second", "time")
    invocation = Unit("invocation", "count")
    allocation = Unit("allocation", "count")
    submission = Unit("submission", "count")
    kernel = Unit("kernel", "count")


@dataclass(frozen=True, slots=True)
class Quantity:
    name: str
    value: ShapeDim
    unit: Unit


@dataclass(frozen=True, slots=True)
class FormulaRef:
    id: str
    version: int

    def __post_init__(self):
        if not self.id or self.version < 1:
            raise ValueError("formula identity requires a name and positive version")


@dataclass(frozen=True, slots=True)
class Port:
    path: tuple[str | int, ...]
    value: int


@dataclass(frozen=True, slots=True)
class FormulaCall:
    occurrence: int
    formula: FormulaRef
    parent: int | None
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    nodes: tuple[int, ...]
    quantities: tuple[Quantity, ...] = ()
    static: tuple[tuple[tuple[str | int, ...], Any], ...] = ()
    complete: bool = True
    metric: str | None = None

    def remap(self, values: dict[int, int], nodes: dict[int, int]) -> FormulaCall:
        return replace(
            self,
            inputs=tuple(Port(port.path, values[port.value]) for port in self.inputs if port.value in values),
            outputs=tuple(Port(port.path, values[port.value]) for port in self.outputs if port.value in values),
            nodes=tuple(nodes[node] for node in self.nodes if node in nodes),
            complete=self.complete and all(port.value in values for port in (*self.inputs, *self.outputs)),
        )


@dataclass(frozen=True, slots=True)
class FormulaIndex:
    calls: tuple[FormulaCall, ...] = ()

    def __iter__(self) -> Iterator[FormulaCall]:
        return iter(self.calls)

    def call(self, occurrence: int) -> FormulaCall:
        if 0 <= occurrence < len(self.calls) and self.calls[occurrence].occurrence == occurrence:
            return self.calls[occurrence]
        for call in self.calls:
            if call.occurrence == occurrence:
                return call
        raise KeyError(occurrence)

    def occurrences(self, definition: Formula) -> tuple[FormulaCall, ...]:
        if not isinstance(definition, Formula):
            raise TypeError("formula lookup requires a Formula object")
        return tuple(call for call in self.calls if call.formula == definition.ref)

    def children(self, occurrence: int | None) -> tuple[FormulaCall, ...]:
        return tuple(call for call in self.calls if call.parent == occurrence)


class Formula:
    def __init__(self, function: Callable, *, id: str, version: int = 1,
                 metric: str | None = None, rows: str | None = None):
        self.ref = FormulaRef(id, version)
        self.function = function
        self.metric, self.rows = metric, rows
        self.signature = inspect.signature(function)
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        from .tensor.tracing import Tensor, active_trace

        state = active_trace()
        arguments = self.signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        flat = tuple(leaves(arguments.arguments))
        inputs = tuple(Port(path, value.value_id) for path, value in flat if isinstance(value, Tensor))
        if any(value._trace is not state for _, value in flat if isinstance(value, Tensor)):
            raise ValueError("formula arguments belong to different traces")
        static = tuple((path, value) for path, value in flat if not isinstance(value, Tensor))
        # Check stable encodability now; never put live providers or preferences in math.
        from .tensor.graph import _stable

        _stable(static)
        occurrence = len(state.formula_calls)
        parent = state.formula_stack[-1] if state.formula_stack else None
        state.formula_calls.append(None)
        state.formula_stack.append(occurrence)
        state.formula_quantities[occurrence] = []
        start = len(state.nodes)
        start_values = len(state.values)
        resource_versions = dict(state._resource_versions)
        try:
            if self.rows is not None:
                quantity("tokens", arguments.arguments[self.rows].shape[0], unit=units.token)
            result = self.function(*args, **kwargs)
            outputs = tuple(Port(path, value.value_id) for path, value in leaves(result)
                            if isinstance(value, Tensor))
            if any(value._trace is not state for _, value in leaves(result) if isinstance(value, Tensor)):
                raise ValueError("formula returned a tensor from another trace")
            boundary = {port.value for port in inputs}
            produced = {output for node in state.nodes[start:] for output in node.outputs}
            captured = {value for node in state.nodes[start:] for value in node.inputs
                        if value not in produced and value not in boundary}
            captured.update(port.value for port in outputs if port.value not in produced and port.value not in boundary)
            if captured:
                raise ValueError("formula tensor dependencies must be explicit arguments, not captured tensors")
            state.formula_calls[occurrence] = FormulaCall(
                occurrence, self.ref, parent, inputs, outputs, tuple(range(start, len(state.nodes))),
                tuple(state.formula_quantities[occurrence]), static, metric=self.metric,
            )
            return result
        except BaseException:
            # A caller may catch a rejected formula and continue tracing. Do not
            # leave orphan nodes, failed occurrences or advanced resource versions
            # in the enclosing mathematical definition.
            del state.nodes[start:]
            del state.values[start_values:]
            del state.formula_calls[occurrence:]
            for key in tuple(state.formula_quantities):
                if key >= occurrence:
                    del state.formula_quantities[key]
            state._resource_versions.clear()
            state._resource_versions.update(resource_versions)
            raise
        finally:
            state.formula_stack.pop()


def formula(function=None, *, id: str | None = None, version: int = 1,
            metric: str | None = None, rows: str | None = None):
    def decorate(body):
        return Formula(body, id=id or body.__name__, version=version, metric=metric, rows=rows)
    return decorate(function) if function is not None else decorate


def quantity(name: str, value: ShapeDim, *, unit: Unit) -> None:
    from .tensor.tracing import active_trace

    state = active_trace()
    if not state.formula_stack:
        raise RuntimeError("quantities must be declared inside a formula")
    current = state.formula_quantities[state.formula_stack[-1]]
    if not name or any(item.name == name for item in current):
        raise ValueError("quantity names must be nonempty and unique within a formula")
    if type(value) is int and value < 0:
        raise ValueError("formula quantities cannot be negative")
    current.append(Quantity(name, value, unit))


def semantic_fingerprint(graph, call: FormulaCall) -> str:
    """Canonical mathematical instance, excluding source, occurrence and value identity."""
    from .tensor.graph import _stable

    if not call.complete:
        raise ValueError("a pruned partial occurrence cannot identify a complete formula")
    ids: dict[int, int] = {}
    resources: dict[int, int] = {}
    versions: dict[int, int] = {}
    def value_id(value):
        if value not in ids:
            ids[value] = len(ids)
        return ids[value]
    def resource(value):
        if value.resource_id is None:
            return None
        identity = value.resource_id
        if identity not in resources:
            resources[identity] = len(resources)
            versions[identity] = value.resource_version or 0
        return resources[identity], (value.resource_version or 0) - versions[identity]

    inputs = [(port.path, value_id(port.value), _stable(graph.value(port.value).spec),
               resource(graph.value(port.value)))
              for port in call.inputs]
    nodes = []
    for node_id in call.nodes:
        node = graph.node(node_id)
        inputs_ids = [value_id(value) for value in node.inputs]
        outputs = [(value_id(value), _stable(graph.value(value).spec), resource(graph.value(value)))
                   for value in node.outputs]
        # Resource IDs are graph-local, so encode operand-relative effects instead.
        from .tensor.primitive import primitives
        primitive = primitives.get(node.operation)
        nodes.append((node.operation, inputs_ids, outputs, _stable(node.attributes),
                      primitive.resource_reads, primitive.resource_writes, primitive.aliases,
                      primitive.host_observation, _stable(primitive.numerical)))
    payload = (_stable(call.formula), inputs, nodes,
               [(port.path, value_id(port.value)) for port in call.outputs],
               _stable(call.static), _stable(call.quantities), call.metric)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class FormulaHandle:
    """A concrete occurrence in one trace, not a string path or implementation ID."""

    graph: Graph
    call: FormulaCall

    def __post_init__(self):
        if self.graph.formulas.call(self.call.occurrence) is not self.call:
            raise ValueError("formula occurrence does not belong to this graph")

    def __hash__(self):
        return hash((id(self.graph), self.call.occurrence))

    def __eq__(self, other):
        return isinstance(other, FormulaHandle) and self.graph is other.graph and self.call is other.call

    @property
    def definition(self) -> FormulaRef:
        return self.call.formula

    @property
    def semantic_identity(self) -> str:
        return semantic_fingerprint(self.graph, self.call)

    @property
    def children(self) -> tuple[FormulaHandle, ...]:
        return tuple(FormulaHandle(self.graph, call)
                     for call in self.graph.formulas.children(self.call.occurrence))

    @property
    def parent(self) -> FormulaHandle | None:
        if self.call.parent is None:
            return None
        return FormulaHandle(self.graph, self.graph.formulas.call(self.call.parent))

    def occurrences(self, definition: Formula) -> tuple[FormulaHandle, ...]:
        """Matching direct children, retaining their actual trace order."""
        if not isinstance(definition, Formula):
            raise TypeError("formula lookup requires a Formula object")
        return tuple(child for child in self.children if child.definition == definition.ref)

    def isolate(self):
        from .isolation import isolate

        return isolate(self)


@dataclass(frozen=True, slots=True)
class FormulaTree:
    """A navigation view over existing occurrences; shared data edges remain a DAG."""

    graph: Graph

    @property
    def roots(self) -> tuple[FormulaHandle, ...]:
        return tuple(FormulaHandle(self.graph, call) for call in self.graph.formulas.children(None))

    def __iter__(self) -> Iterator[FormulaHandle]:
        return (FormulaHandle(self.graph, call) for call in self.graph.formulas)

    def occurrences(self, definition: Formula) -> tuple[FormulaHandle, ...]:
        return tuple(FormulaHandle(self.graph, call) for call in self.graph.formulas.occurrences(definition))

    def affected(self, changed: tuple[FormulaHandle, ...]) -> tuple[FormulaHandle, ...]:
        """Transitive composition and data/state consumers, without summing their times."""
        if any(not isinstance(handle, FormulaHandle) or handle.graph is not self.graph for handle in changed):
            raise ValueError("changed formulas must belong to this trace")
        affected = {handle.call.occurrence for handle in changed}
        changed_values = {port.value for handle in changed for port in handle.call.outputs}
        progress = True
        while progress:
            progress = False
            for call in self.graph.formulas:
                if call.occurrence in affected:
                    if call.parent is not None and call.parent not in affected:
                        affected.add(call.parent)
                        progress = True
                    for port in call.outputs:
                        if port.value not in changed_values:
                            changed_values.add(port.value)
                            progress = True
                    continue
                if any(port.value in changed_values for port in call.inputs):
                    affected.add(call.occurrence)
                    progress = True
            for node in self.graph.nodes:
                if any(value in changed_values for value in node.inputs):
                    for value in node.outputs:
                        if value not in changed_values:
                            changed_values.add(value)
                            progress = True
        return tuple(handle for handle in self if handle.call.occurrence in affected)
