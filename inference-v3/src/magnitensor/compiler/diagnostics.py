"""Structured compiler provenance and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..tensor.graph import Graph
from .lowering import Cover, SubmissionUnit
from .memory import MemoryPlan


@dataclass(frozen=True, slots=True)
class CandidateDiagnostic:
    name: str
    nodes: tuple[int, ...]
    selected: bool
    reason: str | None
    estimated_seconds: float
    workspace_bytes: int
    kernel_count: int


@dataclass(frozen=True, slots=True)
class CompilationDiagnostics:
    graph_name: str
    graph_fingerprint: str
    compiler_identity: str
    capability_fingerprint: str
    mode: str
    precision: str
    candidates: tuple[CandidateDiagnostic, ...]
    materializations: Mapping[int, str]
    arena_bytes: int
    submissions: tuple[tuple[str, ...], ...]
    submission_reasons: tuple[str, ...]
    dispatches: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "materializations", MappingProxyType(dict(self.materializations)))

    def render(self) -> str:
        lines = [
            f"graph {self.graph_name} {self.graph_fingerprint}",
            f"mode={self.mode} precision={self.precision}",
            f"compiler={self.compiler_identity}",
            f"capability={self.capability_fingerprint}",
            f"arena={self.arena_bytes} bytes units={len(self.submissions)} "
            f"kernels={self.dispatches}",
        ]
        for index, (names, reason) in enumerate(
            zip(self.submissions, self.submission_reasons, strict=True)
        ):
            lines.append(f"unit {index}: {', '.join(names)} ({reason})")
        for item in self.candidates:
            status = "selected" if item.selected else f"rejected: {item.reason}"
            lines.append(f"candidate {item.name} nodes={item.nodes} {status}")
        return "\n".join(lines)


def build_diagnostics(
    graph: Graph,
    candidates,
    cover: Cover,
    memory: MemoryPlan,
    submissions: tuple[SubmissionUnit, ...],
    *,
    compiler_identity: str,
    capability_fingerprint: str,
    mode: str,
    precision: str,
) -> CompilationDiagnostics:
    selected = {id(item) for item in cover.candidates}
    items = tuple(
        CandidateDiagnostic(
            candidate.name,
            tuple(sorted(candidate.nodes)),
            id(candidate) in selected,
            None if id(candidate) in selected else cover.rejected.get(candidate.name),
            candidate.estimated_seconds,
            candidate.workspace_bytes,
            candidate.kernel_count,
        )
        for candidate in candidates
    )
    return CompilationDiagnostics(
        graph.name,
        graph.fingerprint,
        compiler_identity,
        capability_fingerprint,
        mode,
        precision,
        items,
        {value: placement.storage.value for value, placement in memory.values.items()},
        memory.arena_bytes,
        tuple(tuple(candidate.name for candidate in unit.candidates) for unit in submissions),
        tuple(unit.reason for unit in submissions),
        sum(unit.kernel_count for unit in submissions),
    )
