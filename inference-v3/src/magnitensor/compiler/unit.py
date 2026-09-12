"""A maximal pre-finalization TileLang compilation unit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .lowering import Candidate, KernelBinding, SubmissionUnit
from .memory import MemoryPlan


class ParameterKind(StrEnum):
    DYNAMIC = "dynamic"
    CONSTANT = "constant"
    RESOURCE = "resource"
    OUTPUT = "output"
    ARENA = "arena"


type BindingKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class UnitParameter:
    index: int
    name: str
    key: BindingKey
    spec: TensorSpec
    kind: ParameterKind


@dataclass(frozen=True, slots=True)
class KernelCall:
    candidate: Candidate
    bindings: tuple[KernelBinding, ...]
    workspace: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TileCompilationUnit:
    """A maximal unit that the private TileLang adapter finalizes exactly once."""

    name: str
    graph_fingerprint: str
    parameters: tuple[UnitParameter, ...]
    calls: tuple[KernelCall, ...]
    output_parameters: tuple[int, ...]

    @property
    def signature(self) -> tuple[TensorSpec, ...]:
        return tuple(parameter.spec for parameter in self.parameters)


def build_unit(graph: Graph, memory: MemoryPlan, unit: SubmissionUnit) -> TileCompilationUnit:
    keys: dict[BindingKey, tuple[str, TensorSpec, ParameterKind]] = {}

    def add_value(value_id: int) -> None:
        placement = memory.values[value_id]
        if placement.source is not None:
            add_value(placement.source)
            return
        kind = ParameterKind(placement.storage.value)
        if kind == ParameterKind.ARENA:
            kind = ParameterKind.ARENA
        value = graph.values[value_id]
        keys[("value", value_id, 0)] = (f"v{value_id}", value.spec, kind)

    for candidate in unit.candidates:
        for value_id in (*candidate.inputs, *candidate.outputs):
            add_value(value_id)

    for candidate in unit.candidates:
        for index, spec in enumerate(candidate.workspace):
            keys[("workspace", min(candidate.nodes), index)] = (
                f"w{min(candidate.nodes)}_{index}",
                spec,
                ParameterKind.ARENA,
            )

    ordered = sorted(
        keys.items(),
        key=lambda item: (
            0 if item[1][2] == ParameterKind.CONSTANT else 1,
            item[0],
        ),
    )
    parameters = tuple(
        UnitParameter(index, name, key, spec, kind)
        for index, (key, (name, spec, kind)) in enumerate(ordered)
    )
    parameter_by_key = {parameter.key: parameter.index for parameter in parameters}

    calls = []
    for candidate in unit.candidates:
        bindings = []
        for value_id in (*candidate.inputs, *candidate.outputs):
            placement = memory.values[value_id]
            source = placement.source if placement.source is not None else value_id
            access = "write" if value_id in candidate.outputs else "read"
            bindings.append(
                KernelBinding(
                    source, parameters[parameter_by_key[("value", source, 0)]].name, access
                )
            )
        workspace = tuple(
            parameter_by_key[("workspace", min(candidate.nodes), index)]
            for index in range(len(candidate.workspace))
        )
        calls.append(KernelCall(candidate, tuple(bindings), workspace))

    output_parameters = tuple(
        parameter.index
        for parameter in parameters
        if parameter.key[0] == "value" and parameter.key[1] in graph.outputs
    )
    return TileCompilationUnit(
        f"mt_{graph.fingerprint[:20]}_{unit.index}",
        graph.fingerprint,
        parameters,
        tuple(calls),
        output_parameters,
    )
