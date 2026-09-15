"""Structured compiler provenance and diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..tensor.graph import Graph
from .lowering import BoundOperation, SubmissionUnit
from .memory import MemoryPlan


@dataclass(frozen=True, slots=True)
class OperationDiagnostic:
    name: str
    nodes: tuple[int, ...]
    workspace_bytes: int
    kernel_count: int


@dataclass(frozen=True, slots=True)
class CompilationDiagnostics:
    graph_name: str
    graph_fingerprint: str
    compiler_identity: str
    configuration_identity: str
    mode: str
    precision: str
    operations: tuple[OperationDiagnostic, ...]
    materializations: Mapping[int, str]
    temporary_bytes: int
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
            f"configuration={self.configuration_identity}",
            f"temporaries={self.temporary_bytes} bytes units={len(self.submissions)} "
            f"numerical-kernel-upper-bound={self.dispatches}",
        ]
        for index, (names, reason) in enumerate(
            zip(self.submissions, self.submission_reasons, strict=True)
        ):
            lines.append(f"unit {index}: {', '.join(names)} ({reason})")
        for item in self.operations:
            lines.append(f"operation {item.name} nodes={item.nodes}")
        return "\n".join(lines)

    def render_summary(self) -> str:
        """Render the defined implementation, not a candidate-selection report."""
        counts = Counter(item.name.split("@", 1)[0] for item in self.operations)
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        return (
            f"graph={self.graph_name} mode={self.mode} nodes="
            f"{sum(len(item.nodes) for item in self.operations)} "
            f"units={len(self.submissions)} numerical-kernel-upper-bound={self.dispatches} "
            f"temporaries={self.temporary_bytes} operations=[{rendered}]"
        )


def build_diagnostics(
    graph: Graph,
    operations: tuple[BoundOperation, ...],
    memory: MemoryPlan,
    submissions: tuple[SubmissionUnit, ...],
    *,
    compiler_identity: str,
    configuration_identity: str,
    mode: str,
    precision: str,
) -> CompilationDiagnostics:
    items = tuple(
        OperationDiagnostic(
            candidate.name,
            tuple(sorted(candidate.nodes)),
            candidate.workspace_bytes,
            candidate.kernel_count,
        )
        for candidate in operations
    )
    return CompilationDiagnostics(
        graph.name,
        graph.fingerprint,
        compiler_identity,
        configuration_identity,
        mode,
        precision,
        items,
        {value: placement.storage.value for value, placement in memory.values.items()},
        memory.temporary_bytes,
        tuple(tuple(operation.name for operation in unit.operations) for unit in submissions),
        tuple(unit.reason for unit in submissions),
        sum(unit.kernel_count for unit in submissions),
    )
