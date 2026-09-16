"""A maximal pre-finalization TileLang compilation unit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .lowering import BoundOperation, KernelBinding, SubmissionUnit
from .memory import MemoryPlan
from .program import define_kernel


class ParameterKind(StrEnum):
    DYNAMIC = "dynamic"
    CONSTANT = "constant"
    RESOURCE = "resource"
    OUTPUT = "output"
    TEMPORARY = "temporary"


type BindingKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class UnitParameter:
    index: int
    name: str
    key: BindingKey
    spec: TensorSpec
    kind: ParameterKind
    offset: bool = True


@dataclass(frozen=True, slots=True)
class KernelCall:
    operation: BoundOperation
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


def build_unit(graph: Graph, memory: MemoryPlan, unit: SubmissionUnit, *,
               bound_offsets: Mapping[int, int] | None = None) -> TileCompilationUnit:
    # A fixed binding proves its origin for the lifetime of the executable.
    # Unbound inputs, resources and constants must accept arbitrary views.
    bound_offsets = bound_offsets or {}
    keys: dict[BindingKey, tuple[str, TensorSpec, ParameterKind]] = {}

    def add_value(value_id: int) -> None:
        placement = memory.values[value_id]
        if placement.source is not None:
            add_value(placement.source)
            return
        kind = ParameterKind(placement.storage.value)
        value = graph.values[value_id]
        keys[("value", value_id, 0)] = (f"v{value_id}", value.spec, kind)

    for candidate in unit.operations:
        for value_id in (*candidate.inputs, *candidate.outputs):
            add_value(value_id)

    for candidate in unit.operations:
        for index, spec in enumerate(candidate.workspace):
            keys[("workspace", min(candidate.nodes), index)] = (
                f"w{min(candidate.nodes)}_{index}",
                spec,
                ParameterKind.TEMPORARY,
            )

    ordered = sorted(
        keys.items(),
        key=lambda item: (
            0 if item[1][2] == ParameterKind.CONSTANT else 1,
            item[0],
        ),
    )
    parameters = tuple(
        UnitParameter(
            index, name, key, spec, kind,
            kind in (ParameterKind.DYNAMIC, ParameterKind.RESOURCE, ParameterKind.CONSTANT)
            and not (key[0] == "value" and bound_offsets.get(key[1]) == 0),
        )
        for index, (key, (name, spec, kind)) in enumerate(ordered)
    )
    parameter_by_key = {parameter.key: parameter.index for parameter in parameters}
    parameter_by_name = {parameter.name: parameter for parameter in parameters}

    calls = []
    for candidate in unit.operations:
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
        if candidate.definition is not None:
            operands = tuple(parameter_by_name[binding.parameter] for binding in bindings) + tuple(
                parameters[index] for index in workspace
            )
            ports = tuple(
                replace(port, offset=parameter.offset)
                for port, parameter in zip(candidate.definition.ports, operands, strict=True)
            )
            if ports != candidate.definition.ports:
                candidate = replace(candidate, definition=define_kernel(candidate.emitter, ports))
        calls.append(KernelCall(candidate, tuple(bindings), workspace))

    output_parameters = tuple(
        parameter.index
        for parameter in parameters
        if parameter.key[0] == "value" and parameter.key[1] in graph.outputs
    )
    return TileCompilationUnit(
        f"ops_{graph.fingerprint[:20]}_{unit.index}",
        graph.fingerprint,
        parameters,
        tuple(calls),
        output_parameters,
    )
