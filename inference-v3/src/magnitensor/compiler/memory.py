"""Graph-derived materialization and temporary arena planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .lowering import Capabilities, Cover


class StorageClass(StrEnum):
    DYNAMIC = "dynamic"
    CONSTANT = "constant"
    RESOURCE = "resource"
    OUTPUT = "output"
    ARENA = "arena"
    ALIAS = "alias"


@dataclass(frozen=True, slots=True)
class Placement:
    storage: StorageClass
    spec: TensorSpec
    offset: int = 0
    source: int | None = None


@dataclass(frozen=True, slots=True)
class WorkspacePlacement:
    candidate: str
    index: int
    spec: TensorSpec
    offset: int


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    values: Mapping[int, Placement]
    workspace: tuple[WorkspacePlacement, ...]
    arena_bytes: int
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


def plan_memory(graph: Graph, cover: Cover, capabilities: Capabilities) -> MemoryPlan:
    candidate_index = {
        node: index for index, candidate in enumerate(cover.candidates) for node in candidate.nodes
    }
    candidate_by_node = {
        node: candidate for candidate in cover.candidates for node in candidate.nodes
    }
    placements: dict[int, Placement] = {}
    for value_id in graph.inputs:
        placements[value_id] = Placement(StorageClass.DYNAMIC, graph.values[value_id].spec)
    for value_id in graph.constants:
        placements[value_id] = Placement(StorageClass.CONSTANT, graph.values[value_id].spec)
    for value_id in graph.resources:
        placements[value_id] = Placement(StorageClass.RESOURCE, graph.values[value_id].spec)

    alias_sources = {
        output: source for candidate in cover.candidates for output, source in candidate.aliases
    }
    output_set = set(graph.outputs)
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
        consumers = [
            candidate_index[node.id]
            for node in graph.nodes
            if value.id in node.inputs and node.id in candidate_index
        ]
        end = max(consumers, default=producer)
        alignment = _alignment(value.spec, capabilities)
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

    for index, candidate in enumerate(cover.candidates):
        for workspace_index, spec in enumerate(candidate.workspace):
            alignment = _alignment(spec, capabilities)
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

    offsets, arena_bytes = _assign(tuple(intervals))
    workspaces = []
    for interval in intervals:
        offset = offsets[interval.identity]
        if interval.identity[0] == "value":
            value_id = interval.identity[1]
            placements[value_id] = Placement(StorageClass.ARENA, interval.spec, offset)
        else:
            candidate_index_value, workspace_index = interval.identity[1:]
            workspaces.append(
                WorkspacePlacement(
                    cover.candidates[candidate_index_value].name,
                    workspace_index,
                    interval.spec,
                    offset,
                )
            )
    alignment = max((item.alignment for item in intervals), default=1)
    return MemoryPlan(placements, tuple(workspaces), arena_bytes, alignment)


def _alignment(spec: TensorSpec, capabilities: Capabilities) -> int:
    return max(spec.dtype.itemsize, capabilities.alignments.get(spec.dtype, spec.dtype.itemsize))


def _assign(intervals: tuple[_Interval, ...]) -> tuple[dict[tuple[str, int, int], int], int]:
    ordered = sorted(intervals, key=lambda item: (item.start, -item.size, item.identity))
    active: list[tuple[int, int, int]] = []  # end, offset, size
    free: list[tuple[int, int]] = []
    offsets = {}
    extent = 0
    for interval in ordered:
        retained = []
        for end, offset, size in active:
            if end < interval.start:
                free.append((offset, size))
            else:
                retained.append((end, offset, size))
        active = retained
        choice = None
        for index, (offset, size) in enumerate(free):
            aligned = _align(offset, interval.alignment)
            if aligned + interval.size <= offset + size:
                waste = size - interval.size
                if choice is None or waste < choice[0]:
                    choice = waste, index, aligned
        if choice is None:
            offset = _align(extent, interval.alignment)
            extent = offset + interval.size
        else:
            _, index, offset = choice
            free_offset, free_size = free.pop(index)
            before = offset - free_offset
            after = free_offset + free_size - (offset + interval.size)
            if before:
                free.append((free_offset, before))
            if after:
                free.append((offset + interval.size, after))
        offsets[interval.identity] = offset
        active.append((interval.end, offset, interval.size))
    return offsets, extent


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
