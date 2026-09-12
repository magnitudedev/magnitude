"""Bounded whole-graph lowering, cover selection, and submission partitioning."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from ..tensor.graph import Graph
from ..tensor.types import DType, Layout, TensorSpec
from .tuning import EMPTY_TUNING, TuningDatabase, TuningKey


@dataclass(frozen=True, slots=True)
class MatrixInstruction:
    m: int
    n: int
    k: int
    input_dtype: DType
    accumulation_dtype: DType


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Backend-neutral target behavior reported by TileLang."""

    subgroup_width: int
    threads_per_group: int
    shared_memory_bytes: int
    matrix_instructions: tuple[MatrixInstruction, ...] = ()
    supported_dtypes: frozenset[DType] = frozenset(DType)
    memory_scopes: frozenset[str] = frozenset({"global", "local"})
    barrier_scopes: frozenset[str] = frozenset()
    asynchronous_copy: bool = False
    subgroup_exchange: bool = False
    vector_bytes: tuple[int, ...] = (1,)
    atomics: frozenset[DType] = frozenset()
    features: frozenset[str] = frozenset()
    alignments: Mapping[DType, int] = field(default_factory=dict)
    native_multi_launch: bool = False
    partial_binding: bool = False
    max_kernels_per_program: int | None = None
    fingerprint: str = "portable"

    def __post_init__(self) -> None:
        if self.subgroup_width <= 0 or self.threads_per_group <= 0 or self.shared_memory_bytes < 0:
            raise ValueError("invalid target capability geometry")
        if self.max_kernels_per_program is not None and self.max_kernels_per_program <= 0:
            raise ValueError("kernel program limit must be positive")
        if (
            not self.supported_dtypes
            or any(value <= 0 for value in self.vector_bytes)
            or any(not value for value in self.features)
        ):
            raise ValueError("target capability sets must not be empty or invalid")
        if not self.fingerprint:
            raise ValueError("capability fingerprint must not be empty")
        object.__setattr__(self, "alignments", MappingProxyType(dict(self.alignments)))


HOST_CAPABILITIES = Capabilities(1, 1, 0, fingerprint="host-reference")


@dataclass(frozen=True, slots=True)
class KernelBinding:
    graph_value: int
    parameter: str
    access: str = "read"


class KernelEmitter(Protocol):
    """Emit selected TileLang work using already-bound tensor parameters."""

    def __call__(self, operands: tuple[Any, ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    nodes: frozenset[int]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    emitter: KernelEmitter
    estimated_seconds: float
    workspace: tuple[TensorSpec, ...] = ()
    accepted_layouts: Mapping[int, tuple[Layout, ...]] = field(default_factory=dict)
    produced_layouts: Mapping[int, Layout] = field(default_factory=dict)
    aliases: tuple[tuple[int, int], ...] = ()
    kernel_count: int = 1
    tuning_key: TuningKey | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.nodes or self.estimated_seconds < 0 or self.kernel_count < 0:
            raise ValueError("invalid lowering candidate")
        object.__setattr__(self, "accepted_layouts", MappingProxyType(dict(self.accepted_layouts)))
        object.__setattr__(self, "produced_layouts", MappingProxyType(dict(self.produced_layouts)))

    @property
    def workspace_bytes(self) -> int:
        return sum(spec.storage_nbytes for spec in self.workspace)


class LoweringRule(Protocol):
    name: str

    def enumerate(
        self, graph: Graph, root: int, context: LoweringContext
    ) -> Iterable[Candidate]: ...


@dataclass(frozen=True, slots=True)
class LoweringContext:
    capabilities: Capabilities
    mode: str
    precision: str
    compiler_identity: str
    workspace_limit: int
    tuning: TuningDatabase = EMPTY_TUNING


class LoweringRegistry:
    def __init__(self, *, max_candidates_per_root: int = 32, max_region_nodes: int = 16):
        self.max_candidates_per_root = max_candidates_per_root
        self.max_region_nodes = max_region_nodes
        self._rules: list[LoweringRule] = []

    def register(self, rule: LoweringRule) -> LoweringRule:
        if any(existing.name == rule.name for existing in self._rules):
            raise ValueError(f"lowering rule {rule.name!r} is already registered")
        self._rules.append(rule)
        return rule

    @property
    def rules(self) -> tuple[LoweringRule, ...]:
        return tuple(self._rules)

    def enumerate(self, graph: Graph, context: LoweringContext) -> tuple[Candidate, ...]:
        unsupported = {
            value.spec.dtype
            for value in graph.values
            if value.spec.dtype not in context.capabilities.supported_dtypes
        }
        if unsupported:
            raise ValueError(
                f"target does not support graph dtypes {sorted(item.value for item in unsupported)}"
            )
        candidates: list[Candidate] = []
        for root in range(len(graph.nodes)):
            count = 0
            for rule in self._rules:
                for candidate in rule.enumerate(graph, root, context):
                    _validate_candidate(graph, candidate, self.max_region_nodes)
                    if candidate.workspace_bytes <= context.workspace_limit:
                        candidates.append(_with_measured_cost(candidate, context.tuning))
                        count += 1
                        if count > self.max_candidates_per_root:
                            raise ValueError(
                                f"candidate enumeration at node {root} exceeded its bound"
                            )
        return tuple(candidates)


lowerings = LoweringRegistry()


@dataclass(frozen=True, slots=True)
class Cover:
    candidates: tuple[Candidate, ...]
    total_seconds: float
    rejected: Mapping[str, str]

    @property
    def by_node(self) -> Mapping[int, Candidate]:
        result = {}
        for candidate in self.candidates:
            for node in candidate.nodes:
                result[node] = candidate
        return MappingProxyType(result)


def select_cover(graph: Graph, candidates: tuple[Candidate, ...]) -> Cover:
    by_start: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_start[min(candidate.nodes)].append(candidate)
    covered = {node for candidate in candidates for node in candidate.nodes}
    missing = [node for node in range(len(graph.nodes)) if node not in covered]
    if missing:
        raise ValueError(f"no legal lowering covers graph nodes {missing}")
    # Lowering regions are contiguous trace intervals. This turns exact cover
    # selection into a bounded shortest path over node positions instead of an
    # exponential subset search over the full graph.
    best: list[tuple[float, int, tuple[Candidate, ...]] | None] = [
        None for _ in range(len(graph.nodes) + 1)
    ]
    best[-1] = (0.0, 0, ())
    for start in reversed(range(len(graph.nodes))):
        choice = None
        for candidate in by_start[start]:
            end = max(candidate.nodes) + 1
            suffix = best[end]
            if suffix is None:
                continue
            selected = (candidate, *suffix[2])
            proposal = (
                candidate.estimated_seconds + suffix[0],
                candidate.priority + suffix[1],
                selected,
            )
            identity = tuple(item.name for item in selected)
            if choice is None or (proposal[0], -proposal[1], identity) < (
                choice[0],
                -choice[1],
                tuple(item.name for item in choice[2]),
            ):
                choice = proposal
        best[start] = choice

    result = best[0]
    if result is None:
        raise ValueError("lowering candidates do not form a complete non-overlapping graph cover")
    selected_names = {candidate.name for candidate in result[2]}
    rejected = {
        candidate.name: "overlapped or costlier than selected cover"
        for candidate in candidates
        if candidate.name not in selected_names
    }
    return Cover(result[2], result[0], MappingProxyType(rejected))


@dataclass(frozen=True, slots=True)
class SubmissionUnit:
    index: int
    candidates: tuple[Candidate, ...]
    reason: str

    @property
    def kernel_count(self) -> int:
        return sum(candidate.kernel_count for candidate in self.candidates)


def plan_submissions(
    graph: Graph, cover: Cover, capabilities: Capabilities
) -> tuple[SubmissionUnit, ...]:
    if (
        sum(candidate.kernel_count for candidate in cover.candidates) > 1
        and not capabilities.native_multi_launch
    ):
        raise ValueError("TileLang target lacks native multi-launch execution")
    if not capabilities.partial_binding and graph.constants:
        raise ValueError("TileLang target lacks generic partial binding")

    units: list[SubmissionUnit] = []
    current: list[Candidate] = []
    current_kernels = 0
    limit = capabilities.max_kernels_per_program or math.inf

    def flush(reason: str) -> None:
        nonlocal current_kernels
        if current:
            units.append(SubmissionUnit(len(units), tuple(current), reason))
            current.clear()
            current_kernels = 0

    for candidate in cover.candidates:
        if candidate.kernel_count == 0:
            continue
        observed = any(graph.nodes[node].effects.host_observation for node in candidate.nodes)
        if current and current_kernels + candidate.kernel_count > limit:
            flush("qualified TileLang compilation-unit kernel limit")
        current.append(candidate)
        current_kernels += candidate.kernel_count
        if observed:
            flush("required host observation")
    flush("maximal terminal submission unit")
    return tuple(units)


def _with_measured_cost(candidate: Candidate, tuning: TuningDatabase) -> Candidate:
    if candidate.tuning_key is None:
        return candidate
    record = tuning.lookup(candidate.tuning_key, candidate.name)
    if record is None:
        return candidate
    from dataclasses import replace

    return replace(candidate, estimated_seconds=record.latency_seconds)


def _validate_candidate(graph: Graph, candidate: Candidate, max_nodes: int) -> None:
    if len(candidate.nodes) > max_nodes or any(
        not 0 <= node < len(graph.nodes) for node in candidate.nodes
    ):
        raise ValueError(f"candidate {candidate.name} has an invalid region")
    if not _connected(graph, candidate.nodes):
        raise ValueError(f"candidate {candidate.name} region is disconnected")
    if candidate.nodes != frozenset(range(min(candidate.nodes), max(candidate.nodes) + 1)):
        raise ValueError(f"candidate {candidate.name} region is not a contiguous trace interval")
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
    # Node ids are topological and the region is a complete id interval, so a
    # dependency path cannot leave the interval and later re-enter it.


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
