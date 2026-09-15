"""Graph-derived materialization and reusable temporary-slot planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .lowering import BoundOperation


class StorageClass(StrEnum):
    DYNAMIC = "dynamic"
    CONSTANT = "constant"
    RESOURCE = "resource"
    OUTPUT = "output"
    TEMPORARY = "temporary"
    ALIAS = "alias"


@dataclass(frozen=True, slots=True)
class Placement:
    storage: StorageClass
    spec: TensorSpec
    slot: int | None = None
    source: int | None = None


@dataclass(frozen=True, slots=True)
class WorkspacePlacement:
    operation: str
    index: int
    spec: TensorSpec
    slot: int


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    values: Mapping[int, Placement]
    workspace: tuple[WorkspacePlacement, ...]
    temporary_bytes: int
    alignment: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class _Interval:
    identity: tuple[str, int, int]
    start: int
    end: int
    size: int
    alignment: int
    spec: TensorSpec


def plan_memory(graph: Graph, operations: tuple[BoundOperation, ...]) -> MemoryPlan:
    candidate_index = {
        node: index for index, candidate in enumerate(operations) for node in candidate.nodes
    }
    candidate_by_node = {
        node: candidate for candidate in operations for node in candidate.nodes
    }
    placements: dict[int, Placement] = {}
    for value_id in graph.inputs:
        placements[value_id] = Placement(StorageClass.DYNAMIC, graph.values[value_id].spec)
    for value_id in graph.constants:
        placements[value_id] = Placement(StorageClass.CONSTANT, graph.values[value_id].spec)
    for value_id in graph.resources:
        placements[value_id] = Placement(StorageClass.RESOURCE, graph.values[value_id].spec)

    alias_sources = {
        output: source for candidate in operations for output, source in candidate.aliases
    }
    for output in tuple(alias_sources):
        source = alias_sources[output]
        visited = {output}
        while source in alias_sources:
            if source in visited:
                raise ValueError("cyclic operation aliases")
            visited.add(source)
            source = alias_sources[source]
        alias_sources[output] = source
    output_set = set(graph.outputs) | {alias_sources[value] for value in graph.outputs if value in alias_sources}
    intervals: list[_Interval] = []

    for value in graph.values:
        if value.id in placements:
            continue
        if value.id in alias_sources:
            placements[value.id] = Placement(
                StorageClass.ALIAS, value.spec, source=alias_sources[value.id]
            )
            continue
        if value.id in output_set:
            placements[value.id] = Placement(StorageClass.OUTPUT, value.spec)
            continue
        if value.producer is None:
            continue
        producer_candidate = candidate_by_node[value.producer]
        if value.id not in producer_candidate.outputs:
            # Interior fused values are registers/shared storage owned by the
            # selected emitter, not globally materialized graph values.
            continue
        producer = candidate_index[value.producer]
        consumers = [index for index, operation in enumerate(operations)
                     if any(alias_sources.get(operand, operand) == value.id for operand in operation.inputs)]
        end = max(consumers, default=producer)
        alignment = _alignment(value.spec)
        intervals.append(
            _Interval(
                ("value", value.id, 0),
                producer,
                end,
                value.spec.storage_nbytes,
                alignment,
                value.spec,
            )
        )

    for index, candidate in enumerate(operations):
        for workspace_index, spec in enumerate(candidate.workspace):
            alignment = _alignment(spec)
            intervals.append(
                _Interval(
                    ("workspace", index, workspace_index),
                    index,
                    index,
                    spec.storage_nbytes,
                    alignment,
                    spec,
                )
            )

    slots, temporary_bytes = _assign_slots(
        tuple(intervals), max((item.alignment for item in intervals), default=1)
    )
    workspaces = []
    for interval in intervals:
        slot = slots[interval.identity]
        if interval.identity[0] == "value":
            value_id = interval.identity[1]
            placements[value_id] = Placement(StorageClass.TEMPORARY, interval.spec, slot)
        else:
            candidate_index_value, workspace_index = interval.identity[1:]
            workspaces.append(
                WorkspacePlacement(
                    operations[candidate_index_value].name,
                    workspace_index,
                    interval.spec,
                    slot,
                )
            )
    alignment = max((item.alignment for item in intervals), default=1)
    return MemoryPlan(placements, tuple(workspaces), temporary_bytes, alignment)


def _alignment(spec: TensorSpec) -> int:
    return spec.dtype.itemsize


def _assign_slots(
    intervals: tuple[_Interval, ...], alignment: int
) -> tuple[dict[tuple[str, int, int], int], int]:
    """Reuse whole native allocations across non-overlapping lifetimes.

    Native bindings require zero-offset buffers. Splitting a freed byte range
    would create another allocation, not a view into the original allocation;
    counting the virtual address extent would then understate physical memory.
    """
    ordered = sorted(intervals, key=lambda item: (item.start, -item.size, item.identity))
    active: list[tuple[int, int, int]] = []  # end, slot, physical capacity
    free: list[tuple[int, int]] = []
    slots = {}
    extent = 0
    for interval in ordered:
        retained = []
        for end, slot, capacity in active:
            if end < interval.start:
                free.append((slot, capacity))
            else:
                retained.append((end, slot, capacity))
        active = retained
        choices = [
            (capacity, index, slot)
            for index, (slot, capacity) in enumerate(free)
            if capacity >= interval.size and slot % interval.alignment == 0
        ]
        if choices:
            capacity, index, slot = min(choices)
            free.pop(index)
        else:
            slot = _align(extent, alignment)
            capacity = _align(interval.size, alignment)
            extent = slot + capacity
        slots[interval.identity] = slot
        active.append((interval.end, slot, capacity))
    return slots, extent


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
