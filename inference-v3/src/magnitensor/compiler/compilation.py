"""Tracing through maximal pre-bound TileLang compilation units."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from ..runtime.resources import Completion, Device, Execution, Resource
from ..tensor.graph import Graph
from ..tensor.tracing import Signature, trace
from ..tensor.types import TensorSpec
from .diagnostics import CompilationDiagnostics, build_diagnostics
from .lowering import LoweringContext, LoweringRegistry, lowerings, plan_submissions, select_cover
from .memory import MemoryPlan, StorageClass, plan_memory
from .tuning import EMPTY_TUNING, TuningDatabase
from .unit import BindingKey, ParameterKind, TileCompilationUnit, build_unit


@dataclass(frozen=True, slots=True)
class CompileOptions:
    mode: str
    precision: str = "model"
    workspace_limit: int | None = None
    tuning: TuningDatabase = EMPTY_TUNING
    dimensions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mode:
            raise ValueError("compile mode must not be empty")
        if self.workspace_limit is not None and self.workspace_limit < 0:
            raise ValueError("workspace limit must not be negative")
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))


@dataclass(slots=True)
class _CompiledUnit:
    unit: TileCompilationUnit
    executable: Any
    entrypoint: Any
    dynamic: tuple[BindingKey, ...]

    def close(self) -> None:
        self.entrypoint.close()
        self.executable.close()


class CompiledFunction:
    """A whole tensor-function specialization with pre-bound native entrypoints."""

    def __init__(
        self,
        device: Device,
        graph: Graph,
        memory: MemoryPlan,
        units: tuple[_CompiledUnit, ...],
        constants: Mapping[int, Resource],
        owned_storage: tuple[Resource, ...],
        static_values: Mapping[int, Resource],
        static_workspace: Mapping[tuple[str, int], Resource],
        diagnostics: CompilationDiagnostics,
    ):
        self.device = device
        self.graph = graph
        self.memory = memory
        self._units = units
        self._constants = {key: value.fork() for key, value in constants.items()}
        self._owned_storage = owned_storage
        self._static_values = dict(static_values)
        self._static_workspace = dict(static_workspace)
        self.diagnostics = diagnostics
        self._closed = False
        self._active = False

    def submit(
        self, *inputs: Resource, resources: Mapping[int | str, Resource] | None = None
    ) -> Execution:
        if self._closed:
            raise RuntimeError("compiled function is closed")
        if self._active:
            raise RuntimeError("compiled function already has an in-flight invocation")
        if len(inputs) != len(self.graph.inputs):
            raise TypeError(
                f"compiled function expects {len(self.graph.inputs)} inputs, got {len(inputs)}"
            )
        values: dict[int, Resource] = {**self._constants, **self._static_values}
        retained: list[Resource] = []
        allocated_outputs: list[Resource] = []
        try:
            for value_id, resource in zip(self.graph.inputs, inputs, strict=True):
                _check_binding(self.device, self.graph.values[value_id].spec, resource)
                values[value_id] = resource
                retained.append(resource.fork())
            supplied = resources or {}
            for value_id in self.graph.resources:
                value = self.graph.values[value_id]
                resource = supplied.get(value_id)
                if resource is None and value.name is not None:
                    resource = supplied.get(value.name)
                if resource is None:
                    raise KeyError(f"missing mutable resource {value.name or value_id}")
                _check_binding(self.device, value.spec, resource)
                values[value_id] = resource
                retained.append(resource.fork())

            for value_id, placement in self.memory.values.items():
                if placement.storage == StorageClass.OUTPUT:
                    values[value_id] = self.device.allocate(placement.spec)
                    allocated_outputs.append(values[value_id])
                    retained.append(values[value_id].fork())
                elif placement.storage == StorageClass.ALIAS:
                    assert placement.source is not None
                    values[value_id] = values[placement.source]

            native_completions = []
            for unit in self._units:
                dynamic = []
                for key in unit.dynamic:
                    if key[0] == "value":
                        dynamic.append(values[key[1]].native)
                    else:
                        candidate = next(
                            item.candidate.name
                            for item in unit.unit.calls
                            if min(item.candidate.nodes) == key[1]
                        )
                        dynamic.append(self._static_workspace[(candidate, key[2])].native)
                native_completions.append(unit.entrypoint.submit(tuple(dynamic)))
            native = self.device.runtime.join(tuple(native_completions))
            outputs = tuple(values[value_id].fork() for value_id in self.graph.outputs)
            for resource in allocated_outputs:
                resource.close()
            self._active = True
            return Execution(outputs, Completion(native, tuple(retained), self._release_invocation))
        except BaseException:
            for resource in reversed(retained):
                resource.close()
            for resource in allocated_outputs:
                resource.close()
            self._active = False
            raise

    def _release_invocation(self) -> None:
        self._active = False

    def close(self) -> None:
        if self._closed:
            return
        if self._active:
            raise RuntimeError("cannot close a compiled function with an in-flight invocation")
        for unit in reversed(self._units):
            unit.close()
        for resource in reversed(self._owned_storage):
            resource.close()
        for resource in self._constants.values():
            resource.close()
        self._closed = True


def compile(
    function,
    *,
    signature: Signature,
    device: Device,
    constants: Mapping[int | str, Resource],
    options: CompileOptions,
    registry: LoweringRegistry = lowerings,
) -> CompiledFunction:
    graph = trace(function, _specialize_signature(signature, options.dimensions))
    bound_constants = _bind_constants(graph, constants, device)
    workspace_limit = (
        device.available_bytes if options.workspace_limit is None else options.workspace_limit
    )
    context = LoweringContext(
        device.capabilities,
        options.mode,
        options.precision,
        device.compiler_identity,
        workspace_limit,
        options.tuning,
    )
    candidates = registry.enumerate(graph, context)
    cover = select_cover(graph, candidates)
    memory = plan_memory(graph, cover, device.capabilities)
    submissions = plan_submissions(graph, cover, device.capabilities)
    units = []
    owned_storage, static_values, static_workspace = _allocate_static_arena(device, memory)
    try:
        for submission in submissions:
            compilation_unit = build_unit(graph, memory, submission)
            static = {}
            for parameter in compilation_unit.parameters:
                if parameter.kind == ParameterKind.CONSTANT:
                    static[parameter.index] = bound_constants[parameter.key[1]].native
                elif parameter.kind == ParameterKind.ARENA:
                    if parameter.key[0] == "value":
                        static[parameter.index] = static_values[parameter.key[1]].native
                    else:
                        candidate = next(
                            item.candidate.name
                            for item in compilation_unit.calls
                            if min(item.candidate.nodes) == parameter.key[1]
                        )
                        static[parameter.index] = static_workspace[
                            (candidate, parameter.key[2])
                        ].native
            dynamic_indices = tuple(
                parameter.index
                for parameter in compilation_unit.parameters
                if parameter.kind not in (ParameterKind.CONSTANT, ParameterKind.ARENA)
            )
            executable = device.runtime.compile(compilation_unit, compilation_unit.signature)
            try:
                entrypoint = executable.bind(static, dynamic_indices)
            except BaseException:
                executable.close()
                raise
            dynamic = tuple(
                parameter.key
                for parameter in compilation_unit.parameters
                if parameter.kind not in (ParameterKind.CONSTANT, ParameterKind.ARENA)
            )
            units.append(_CompiledUnit(compilation_unit, executable, entrypoint, dynamic))
    except BaseException:
        for unit in reversed(units):
            unit.close()
        for resource in reversed(owned_storage):
            resource.close()
        raise
    diagnostics = build_diagnostics(
        graph,
        candidates,
        cover,
        memory,
        submissions,
        compiler_identity=device.compiler_identity,
        capability_fingerprint=device.capabilities.fingerprint,
        mode=options.mode,
        precision=options.precision,
    )
    return CompiledFunction(
        device,
        graph,
        memory,
        tuple(units),
        bound_constants,
        owned_storage,
        static_values,
        static_workspace,
        diagnostics,
    )


def _allocate_static_arena(device: Device, memory: MemoryPlan):
    arena = device.allocate_arena(memory.arena_bytes, memory.alignment)
    if arena is None:
        return (), {}, {}
    owned = [arena]
    values = {}
    workspaces = {}
    try:
        for value_id, placement in memory.values.items():
            if placement.storage == StorageClass.ARENA:
                view = arena.view(placement.spec, placement.offset)
                values[value_id] = view
                owned.append(view)
        for placement in memory.workspace:
            view = arena.view(placement.spec, placement.offset)
            workspaces[(placement.candidate, placement.index)] = view
            owned.append(view)
        return tuple(owned), values, workspaces
    except BaseException:
        for resource in reversed(owned):
            resource.close()
        raise


def _bind_constants(
    graph: Graph, supplied: Mapping[int | str, Resource], device: Device
) -> dict[int, Resource]:
    result = {}
    for value_id in graph.constants:
        value = graph.values[value_id]
        resource = supplied.get(value_id)
        if resource is None and value.name is not None:
            resource = supplied.get(value.name)
        if resource is None:
            raise KeyError(f"missing immutable constant {value.name or value_id}")
        _check_binding(device, value.spec, resource)
        result[value_id] = resource
    return result


def _specialize_signature(signature: Signature, dimensions: Mapping[str, int]) -> Signature:
    def bind(argument):
        return replace(argument, spec=argument.spec.bind(dict(dimensions)))

    try:
        return Signature(
            tuple(bind(argument) for argument in signature.args),
            {name: bind(argument) for name, argument in signature.kwargs.items()},
            signature.static_kwargs,
        )
    except KeyError as error:
        raise ValueError(
            f"missing specialization for symbolic dimension {error.args[0]!r}"
        ) from error


def _check_binding(device: Device, expected: TensorSpec, resource: Resource) -> None:
    if resource.device is not device:
        raise ValueError("resource belongs to another device")
    if resource.spec != expected:
        raise ValueError(f"resource specification {resource.spec!r} differs from {expected!r}")
