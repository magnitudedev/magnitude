"""Physical operation boundaries, readiness and native submission partitioning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from ..tensor.graph import Graph
from ..tensor.types import Layout, TensorSpec
from .program import KernelDefinition
from .dependencies import CodeDependency
from ..binding import Binding


@dataclass(frozen=True, slots=True)
class CompilerTarget:
    """Resolved physical resources and compiler configuration identity."""

    subgroup_width: int
    threads_per_group: int
    shared_memory_bytes: int
    reference_schedules: bool = False
    identity: str = "host-reference"

    def __post_init__(self) -> None:
        if self.subgroup_width <= 0 or self.threads_per_group <= 0 or self.shared_memory_bytes < 0:
            raise ValueError("invalid compilation resource limits")
        if not self.identity:
            raise ValueError("compiler configuration identity must not be empty")


HOST_TARGET = CompilerTarget(1, 1, 0)


@dataclass(frozen=True, slots=True)
class KernelBinding:
    graph_value: int
    parameter: str
    access: str = "read"


class KernelEmitter(Protocol):
    """Emit selected TileLang work using already-bound tensor parameters."""

    def __call__(self, operands: tuple[Any, ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class BoundOperation:
    name: str
    nodes: frozenset[int]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    emitter: KernelEmitter
    workspace: tuple[TensorSpec, ...] = ()
    accepted_layouts: Mapping[int, tuple[Layout, ...]] = field(default_factory=dict)
    produced_layouts: Mapping[int, Layout] = field(default_factory=dict)
    aliases: tuple[tuple[int, int], ...] = ()
    # Numerical launch count for fixed bodies; upper bound for data-dependent
    # source loops. Import/conversion launches belong to runtime observations.
    kernel_count: int = 1
    definition: KernelDefinition | None = field(default=None, compare=False, repr=False)
    source_loop: Any | None = field(default=None, compare=False, repr=False)
    dependencies: tuple[CodeDependency, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name or not self.nodes or self.kernel_count < 0:
            raise ValueError("invalid physical operation")
        object.__setattr__(self, "accepted_layouts", MappingProxyType(dict(self.accepted_layouts)))
        object.__setattr__(self, "produced_layouts", MappingProxyType(dict(self.produced_layouts)))

    @property
    def workspace_bytes(self) -> int:
        return sum(spec.storage_nbytes for spec in self.workspace)

@dataclass(frozen=True, slots=True)
class LoweringContext:
    compiler_target: CompilerTarget
    mode: str
    precision: str
    compiler_identity: str
    workspace_limit: int
    bindings: Mapping[int, Binding] = field(default_factory=dict)


def order_operations(graph: Graph, operations: tuple[BoundOperation, ...]) -> tuple[BoundOperation, ...]:
    """Order an already-defined implementation; never choose between alternatives."""
    owners = {}
    for index, operation in enumerate(operations):
        _validate_operation(graph, operation)
        for node in operation.nodes:
            if node in owners:
                raise ValueError(f"two physical operations implement node {node}")
            owners[node] = index
        for value, accepted in operation.accepted_layouts.items():
            if graph.value(value).spec.layout not in accepted:
                raise ValueError(f"{operation.name} does not accept the declared layout of {value}")
        for value, layout in operation.produced_layouts.items():
            if graph.value(value).spec.layout != layout:
                raise ValueError(f"{operation.name} changes a declared output layout")
        for output, source in operation.aliases:
            if output not in operation.outputs or source not in operation.inputs:
                raise ValueError("operation alias is not a declared boundary port")
            if graph.value(output).spec.storage_nbytes > graph.value(source).spec.storage_nbytes:
                raise ValueError("operation alias exceeds source backing")
    if missing := set(range(len(graph.nodes))) - owners.keys():
        raise ValueError(f"no physical operation implements nodes {sorted(missing)}")
    predecessors = [set() for _ in operations]
    readers: dict[int, set[int]] = {}
    writers: dict[int, int] = {}
    for node in graph.nodes:
        owner = owners[node.id]
        for value in node.inputs:
            producer = graph.value(value).producer
            if producer is not None and owners[producer] != owner:
                predecessors[owner].add(owners[producer])
        for resource in node.effects.reads:
            if resource in writers and writers[resource] != owner:
                predecessors[owner].add(writers[resource])
            readers.setdefault(resource, set()).add(owner)
        for resource, _, _ in node.effects.writes:
            if resource in writers and writers[resource] != owner:
                predecessors[owner].add(writers[resource])
            predecessors[owner].update(readers.pop(resource, set()) - {owner})
            writers[resource] = owner
    ordered = []
    pending = set(range(len(operations)))
    while pending:
        ready = sorted((index for index in pending if not predecessors[index] & pending),
                       key=lambda index: min(operations[index].nodes))
        if not ready:
            raise ValueError("physical operation grouping creates a data/state dependency cycle")
        ordered.extend(operations[index] for index in ready)
        pending.difference_update(ready)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class SubmissionUnit:
    index: int
    operations: tuple[BoundOperation, ...]
    reason: str

    @property
    def kernel_count(self) -> int:
        return sum(operation.kernel_count for operation in self.operations)


def plan_submissions(
    graph: Graph, operations: tuple[BoundOperation, ...], *, streamed: frozenset[int] = frozenset()
) -> tuple[SubmissionUnit, ...]:
    units: list[SubmissionUnit] = []
    current: list[BoundOperation] = []
    current_kernels = 0
    current_sources: frozenset[int] = frozenset()

    def flush(reason: str) -> None:
        nonlocal current_kernels
        if current:
            units.append(SubmissionUnit(len(units), tuple(current), reason))
            current.clear()
            current_kernels = 0

    for candidate in operations:
        if candidate.kernel_count == 0:
            continue
        if candidate.source_loop is not None:
            flush("source-tiled operation boundary")
            current.append(candidate)
            current_kernels = candidate.kernel_count
            flush("source-tiled operation completion")
            continue
        observed = any(graph.nodes[node].effects.host_observation for node in candidate.nodes)
        sources = frozenset(candidate.inputs) & streamed
        if current and sources != current_sources:
            flush("source residency/transfer boundary")
        current_sources = sources
        current.append(candidate)
        current_kernels += candidate.kernel_count
        if observed:
            flush("required host observation")
    flush("maximal terminal submission unit")
    return tuple(units)


def _validate_operation(graph: Graph, candidate: BoundOperation) -> None:
    if any(
        not 0 <= node < len(graph.nodes) for node in candidate.nodes
    ):
        raise ValueError(f"candidate {candidate.name} has an invalid region")
    if not _connected(graph, candidate.nodes):
        raise ValueError(f"candidate {candidate.name} region is disconnected")
    if not _convex(graph, candidate.nodes):
        raise ValueError(f"candidate {candidate.name} region is not graph-convex")
    internal_outputs = {value for node in candidate.nodes for value in graph.nodes[node].outputs}
    expected_inputs = {
        value
        for node in candidate.nodes
        for value in graph.nodes[node].inputs
        if graph.values[value].producer not in candidate.nodes
    }
    expected_outputs = {
        value
        for value in internal_outputs
        if value in graph.outputs
        or any(consumer not in candidate.nodes for consumer in graph.users[value])
    }
    declared_outputs = set(candidate.outputs)
    if (
        set(candidate.inputs) != expected_inputs
        or not expected_outputs <= declared_outputs
        or not declared_outputs <= internal_outputs
    ):
        raise ValueError(f"candidate {candidate.name} declares incorrect region boundaries")
    aliases = dict(candidate.aliases)
    if len(aliases) != len(candidate.aliases):
        raise ValueError("an operation output cannot have multiple backing aliases")
    for output, source in aliases.items():
        if (output not in candidate.outputs or source not in candidate.inputs or
                graph.alias_root(output) != graph.alias_root(source)):
            raise ValueError("operation aliases disagree with the formula's backing contract")
    input_roots = {graph.alias_root(value) for value in candidate.inputs}
    for output in candidate.outputs:
        if graph.alias_root(output) in input_roots and output not in aliases:
            raise ValueError("operation must preserve the formula's input-backed output alias")


def _convex(graph: Graph, nodes: frozenset[int]) -> bool:
    outside = {user for node in nodes for value in graph.node(node).outputs
               for user in graph.users[value] if user not in nodes}
    visited = set(outside)
    stack = list(outside)
    while stack:
        node = stack.pop()
        for value in graph.node(node).outputs:
            for user in graph.users[value]:
                if user in nodes:
                    return False
                if user not in visited:
                    visited.add(user)
                    stack.append(user)
    return True


def _connected(graph: Graph, nodes: frozenset[int]) -> bool:
    if len(nodes) == 1:
        return True
    adjacency: dict[int, set[int]] = {node: set() for node in nodes}
    consumers: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        for value in graph.nodes[node].inputs:
            consumers[value].append(node)
            producer = graph.values[value].producer
            if producer in nodes:
                adjacency[node].add(producer)
                adjacency[producer].add(node)
    # Values are hyperedges: sibling operations consuming the same external
    # activation are connected even when neither produces the other.
    for related in consumers.values():
        if len(related) > 1:
            anchor = related[0]
            for node in related[1:]:
                adjacency[anchor].add(node)
                adjacency[node].add(anchor)
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == set(nodes)
