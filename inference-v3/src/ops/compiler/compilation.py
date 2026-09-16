"""Tracing through maximal pre-bound TileLang compilation units."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from sys import maxsize
from types import MappingProxyType
from typing import Any

from ..binding import Binding, Residency
from ..formula import FormulaTree
from ..runtime.configuration import DeviceConfiguration
from ..runtime.resources import Completion, DeviceRuntime, Execution, Resource
from ..tensor.graph import Graph, prune_dead_nodes
from ..tensor.tracing import Signature, trace
from ..tensor.types import TensorSpec
from .diagnostics import CompilationDiagnostics, build_diagnostics
from .execution import ExecutionGraph, execution_graph
from .lowering import (
    BoundOperation,
    CompilerTarget,
    LoweringContext,
    SubmissionUnit,
    plan_submissions,
)
from .memory import MemoryPlan, StorageClass, plan_memory
from .schedules import ScheduleResolver
from .unit import BindingKey, ParameterKind, TileCompilationUnit, build_unit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompileOptions:
    mode: str
    precision: str = "model"
    workspace_limit: int | None = None
    dimensions: Mapping[str, int] = field(default_factory=dict)
    schedules: ScheduleResolver | None = None

    def __post_init__(self) -> None:
        if not self.mode:
            raise ValueError("compile mode must not be empty")
        if self.workspace_limit is not None and self.workspace_limit < 0:
            raise ValueError("workspace limit must not be negative")
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))


@dataclass(frozen=True, slots=True)
class CompilationPlan:
    """Pure whole-graph plan produced before allocation or backend compilation."""

    graph: Graph
    operations: tuple[BoundOperation, ...]
    memory: MemoryPlan
    submissions: tuple[SubmissionUnit, ...]
    diagnostics: CompilationDiagnostics
    bindings: Mapping[int, Binding] = field(default_factory=dict)
    execution: ExecutionGraph | None = None
    device_fingerprint: str | None = None
    formula_graph: Graph | None = None

    def __post_init__(self):
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    @property
    def plan(self):
        return self

    @property
    def formulas(self):
        return FormulaTree(self.formula_graph if self.formula_graph is not None else self.graph)


@dataclass(slots=True)
class _CompiledUnit:
    unit: TileCompilationUnit
    executable: Any
    entrypoint: Any
    dynamic: tuple[BindingKey, ...]
    source_loop: Any | None = None

    def close(self) -> None:
        if self.source_loop is not None:
            self.source_loop.close()
        else:
            self.entrypoint.close()
            self.executable.close()


class CompiledFunction:
    """A whole tensor-function specialization with pre-bound native entrypoints."""

    def __init__(
        self,
        device: DeviceRuntime,
        graph: Graph,
        memory: MemoryPlan,
        units: tuple[_CompiledUnit, ...],
        constants: Mapping[int, Resource],
        static_resources: Mapping[int, Resource],
        owned_storage: tuple[Resource, ...],
        static_values: Mapping[int, Resource],
        static_workspace: Mapping[tuple[str, int], Resource],
        diagnostics: CompilationDiagnostics,
        source_bindings: Mapping[int, Binding] | None = None,
        execution_graph: ExecutionGraph | None = None,
        formula_graph: Graph | None = None,
    ):
        self.device = device
        self.graph = graph
        self.formulas = FormulaTree(formula_graph if formula_graph is not None else graph)
        self.memory = memory
        self._invocation_placements = tuple(
            (identity, placement) for identity, placement in memory.values.items()
            if placement.storage in (StorageClass.OUTPUT, StorageClass.ALIAS)
        )
        self._units = units
        self._constants = {}
        self._static_resources = {}
        try:
            for key, value in constants.items():
                self._constants[key] = value.fork()
            for key, value in static_resources.items():
                self._static_resources[key] = value.fork()
        except BaseException:
            for value in (*self._constants.values(), *self._static_resources.values()):
                value.close()
            raise
        self._owned_storage = owned_storage
        self._static_values = dict(static_values)
        self._static_workspace = dict(static_workspace)
        self.diagnostics = diagnostics
        self._source_bindings = dict(source_bindings or {})
        self.execution_graph = execution_graph
        self._closed = False
        self._active = False
        self._output_frame_limit = 0
        self._output_frames: list[dict[int, Resource]] = []
        device._executables.add(self)

    def reuse_output_storage(self, *, max_frames: int = 2) -> None:
        """Retain a bounded set of output backings for sequential invocations.

        Escaped results and checkpoints keep their ordinary leases. A frame is
        reusable only when this compiled owner holds every remaining lease.
        If all retained frames are pinned, the invocation uses fresh outputs.
        """
        self.device.check()
        if self._closed or self._active or self._output_frames:
            raise RuntimeError("output reuse must be configured before invocation")
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("output reuse requires a positive frame bound")
        self._output_frame_limit = max_frames

    def _output_frame(self) -> dict[int, Resource] | None:
        if not self._output_frame_limit:
            return None
        for frame in self._output_frames:
            if all(resource.sole_owner for resource in frame.values()):
                return frame
        if len(self._output_frames) == self._output_frame_limit:
            return None
        frame = {}
        try:
            for identity, placement in self._invocation_placements:
                if placement.storage == StorageClass.OUTPUT:
                    frame[identity] = self.device.allocate(placement.spec)
        except BaseException:
            for resource in frame.values():
                resource.close()
            raise
        self._output_frames.append(frame)
        return frame

    def release_output_storage(self) -> int:
        """Drop cached ownership; escaped results and completion pins survive."""
        self.device.check()
        before = self.device.allocated_bytes
        for frame in self._output_frames:
            for resource in frame.values():
                resource.close()
        self._output_frames.clear()
        return before - self.device.allocated_bytes

    def submit(
        self, *inputs: Resource, resources: Mapping[int | str, Resource] | None = None,
        sources: Mapping[int, Binding] | None = None,
    ) -> Execution:
        """Invoke with optional same-geometry immutable streamed source ports.

        Source rebinding changes values/provenance, never the operation, conversion
        geometry or native code. This supports bounded subregions such as experts
        without compiling or caching a program for every source address.
        """
        if self._closed:
            raise RuntimeError("compiled function is closed")
        if self._active:
            raise RuntimeError("compiled function already has an in-flight invocation")
        if len(inputs) != len(self.graph.inputs):
            raise TypeError(
                f"compiled function expects {len(self.graph.inputs)} inputs, got {len(inputs)}"
            )
        self.device.observations.program(self.graph, self._units)
        source_bindings = dict(self._source_bindings)
        for identity, binding in (sources or {}).items():
            original = source_bindings.get(identity)
            if (original is None or binding.resource is not None or
                    binding.residency != Residency.STREAMED or
                    binding.spec != original.spec or binding.recipe != original.recipe or
                    tuple((plane.group_elements, plane.group_bytes, plane.span.length) for plane in binding.planes) !=
                    tuple((plane.group_elements, plane.group_bytes, plane.span.length) for plane in original.planes)):
                raise ValueError("source invocation changes the prepared streamed port geometry")
            source_bindings[identity] = binding
        values: dict[int, Resource] = {
            **self._constants,
            **self._static_resources,
            **self._static_values,
        }
        retained: list[Resource] = []
        allocated_outputs: list[Resource] = []
        native_completions = []
        prior_submissions = frozenset(self.device._submissions)
        try:
            for value_id, resource in zip(self.graph.inputs, inputs, strict=True):
                _check_binding(self.device, self.graph.values[value_id].spec, resource)
                values[value_id] = resource
                retained.append(resource.fork())
            supplied = resources or {}
            for value_id in self.graph.resources:
                if value_id in self._static_resources:
                    continue
                value = self.graph.values[value_id]
                resource = supplied.get(value_id)
                if resource is None and value.name is not None:
                    resource = supplied.get(value.name)
                if resource is None:
                    raise KeyError(f"missing mutable resource {value.name or value_id}")
                _check_binding(self.device, value.spec, resource)
                values[value_id] = resource
                retained.append(resource.fork())

            output_frame = self._output_frame()
            for value_id, placement in self._invocation_placements:
                if placement.storage == StorageClass.OUTPUT:
                    values[value_id] = (output_frame[value_id].fork() if output_frame is not None
                                        else self.device.allocate(placement.spec))
                    allocated_outputs.append(values[value_id])
                    retained.append(values[value_id].fork())
                elif placement.storage == StorageClass.ALIAS:
                    assert placement.source is not None
                    view = values[placement.source].view(placement.spec)
                    values[value_id] = view
                    retained.append(view)

            for unit_index, unit in enumerate(self._units):
                unit_sources = (self.execution_graph.streamed_by_unit[unit_index]
                           if self.execution_graph is not None else ())
                streamed = []
                if unit_sources and native_completions:
                    native_completions[-1].wait()
                try:
                    for value in unit_sources:
                        from ..runtime.imports import plan_import

                        importer = self.device.prepare_binding(self._source_bindings[value])
                        resource = importer.load(plan=plan_import(source_bindings[value]))
                        values[value] = resource
                        streamed.append(resource)
                    if unit.source_loop is not None:
                        if native_completions:
                            native_completions[-1].wait()
                        native_completions.append(unit.source_loop.submit(values, sources=source_bindings))
                        continue
                    dynamic = []
                    for key in unit.dynamic:
                        if key[0] == "value":
                            dynamic.append(values[key[1]].native)
                        else:
                            candidate = next(
                                item.operation.name
                                for item in unit.unit.calls
                                if min(item.operation.nodes) == key[1]
                            )
                            dynamic.append(self._static_workspace[(candidate, key[2])].native)
                    native = self.device.submit_native(unit.entrypoint, tuple(dynamic))
                    native_completions.append(native)
                    if streamed:
                        native.wait()
                except BaseException:
                    # Do not release source storage while an earlier submission
                    # may still consume it. The failure completion below owns it.
                    retained.extend(streamed)
                    streamed.clear()
                    raise
                finally:
                    for resource in reversed(streamed):
                        resource.close()
            native = self.device.join(tuple(native_completions))
            outputs = tuple(values[value_id].fork() for value_id in self.graph.outputs)
            for resource in allocated_outputs:
                resource.close()
            self._active = True
            return Execution(
                outputs,
                Completion(self.device, native, tuple(retained), self._release_invocation),
            )
        except BaseException:
            # Include work submitted by a composed source/import operation that
            # failed before returning its enclosing completion.
            native_completions.extend(
                native for identity, native in self.device._submissions.items()
                if identity not in prior_submissions and native not in native_completions
            )
            if native_completions:
                self._active = True
                cleanup = Completion(
                    self.device, self.device.join(tuple(native_completions)),
                    tuple(retained) + tuple(allocated_outputs), self._release_invocation,
                )
                # If wait itself fails, the runtime retains this completion and
                # its resources for draining; no in-flight buffer is reclaimed.
                cleanup.wait()
            else:
                for resource in reversed(retained):
                    resource.close()
                for resource in allocated_outputs:
                    resource.close()
                self._active = False
            raise

    def _release_invocation(self) -> None:
        self._active = False

    @property
    def code_dependencies(self):
        dependencies = {}
        for unit in self._units:
            for call in unit.unit.calls:
                operation = call.operation
                authored = list(operation.dependencies)
                if operation.definition is not None:
                    authored.extend(operation.definition.dependencies)
                if operation.source_loop is not None:
                    for template in operation.source_loop.templates:
                        authored.extend(template.definition.dependencies)
                for item in authored:
                    dependencies[item.module, item.symbol] = item
            if unit.source_loop is not None:
                for program in unit.source_loop.programs:
                    for item in program.code_dependencies:
                        dependencies[item.module, item.symbol] = item
                for _, _, importer in unit.source_loop._imports:
                    for program in importer._programs.values():
                        for item in program.code_dependencies:
                            dependencies[item.module, item.symbol] = item
        return tuple(dependencies[key] for key in sorted(dependencies))

    def evidence(self) -> dict[str, str]:
        """Supported compiled sources, never estimates of native instruction counts."""
        result = {}
        for index, unit in enumerate(self._units):
            describe = getattr(unit.executable, "evidence", None)
            if describe is not None:
                for kind, source in describe().items():
                    result[f"unit-{index}/{kind}"] = source
            for call in unit.unit.calls:
                definition = call.operation.definition
                if definition is not None:
                    result[f"{definition.identity}/authored"] = str(definition.program)
        return result

    def close(self) -> None:
        if self._closed:
            return
        if self._active:
            raise RuntimeError("cannot close a compiled function with an in-flight invocation")
        self.device._executables.discard(self)
        for unit in reversed(self._units):
            unit.close()
        for resource in reversed(self._owned_storage):
            resource.close()
        for resource in self._constants.values():
            resource.close()
        for resource in self._static_resources.values():
            resource.close()
        for frame in self._output_frames:
            for resource in frame.values():
                resource.close()
        self._output_frames.clear()
        self._closed = True


def compile(
    function,
    *,
    signature: Signature,
    device: DeviceRuntime,
    constants: Mapping[int | str, Resource | Binding],
    static_resources: Mapping[int | str, Resource] | None = None,
    options: CompileOptions,
) -> CompiledFunction:
    if options.schedules is None and device.schedules is not None:
        options = replace(options, schedules=device.schedules)
    plan = analyze(
        function,
        signature=signature,
        compiler_target=device.compiler_target,
        compiler_identity=device.compiler_identity,
        device_identity=device.evidence_identity if options.schedules is not None else None,
        available_bytes=device.available_bytes,
        constants=constants,
        options=options,
    )
    return materialize(
        plan,
        device=device,
        constants=constants,
        static_resources=static_resources or {},
    )


def analyze(
    function,
    *,
    signature: Signature,
    compiler_target: CompilerTarget | None = None,
    options: CompileOptions,
    compiler_identity: str = "analysis",
    available_bytes: int | None = None,
    constants: Mapping[int | str, Resource | Binding] | None = None,
    device: DeviceConfiguration | None = None,
    device_identity: str | None = None,
) -> CompilationPlan:
    """Plan a function without allocation, code generation, or native execution."""
    graph = trace(function, _specialize_signature(signature, options.dimensions))
    return analyze_graph(
        graph, compiler_target=compiler_target, options=options, compiler_identity=compiler_identity,
        available_bytes=available_bytes, constants=constants, device=device, device_identity=device_identity,
    )


def analyze_graph(
    formula_graph: Graph,
    *,
    compiler_target: CompilerTarget | None = None,
    options: CompileOptions,
    compiler_identity: str = "analysis",
    available_bytes: int | None = None,
    constants: Mapping[int | str, Resource | Binding] | None = None,
    device: DeviceConfiguration | None = None,
    device_identity: str | None = None,
) -> CompilationPlan:
    """The same production planning path for full traces and isolated formulas."""
    if device is not None:
        from ..runtime.tilelang import describe_configuration

        selected_target, selected_compiler = describe_configuration(device)
        if compiler_target is not None and compiler_target != selected_target:
            raise ValueError("explicit compiler target disagrees with the device plan")
        compiler_target, compiler_identity = selected_target, selected_compiler
        if available_bytes is None:
            domains = frozenset(device.selected_endpoints[0].modes[0].domains)
            available_bytes = min(constraint.maximum_bytes for constraint in device.memory_constraints
                                  if domains.intersection(constraint.domains))
    if compiler_target is None:
        raise TypeError("analysis requires a device plan or an explicit compiler target")
    graph = prune_dead_nodes(formula_graph)
    bindings = {}
    for value_id in graph.constants:
        value = graph.value(value_id)
        supplied = (constants or {}).get(value_id, (constants or {}).get(value.name))
        if isinstance(supplied, Binding):
            if supplied.spec != value.spec:
                raise ValueError(f"source binding for {value.name!r} disagrees with formula geometry")
            bindings[value_id] = supplied
    workspace_limit = options.workspace_limit
    if workspace_limit is None:
        workspace_limit = maxsize if available_bytes is None else available_bytes
    if device_identity is None:
        device_identity = device.fingerprint if device is not None else compiler_target.identity
    if options.schedules is not None and device is not None:
        from ..runtime.tilelang import describe_schedule_device

        device_identity = describe_schedule_device(device)
    context = LoweringContext(
        compiler_target,
        options.mode,
        options.precision,
        compiler_identity,
        workspace_limit,
        bindings,
        options.schedules,
        device_identity,
    )
    from ..operation import build_operations

    operations = build_operations(graph, context)
    memory = plan_memory(graph, operations)
    required = memory.temporary_bytes + sum(
        placement.spec.storage_nbytes for placement in memory.values.values()
        if placement.storage == StorageClass.OUTPUT
    ) + max((item.source_loop.peak_bytes for item in operations if item.source_loop is not None), default=0)
    if available_bytes is not None and required > available_bytes:
        raise ValueError(f"operation storage requires {required} bytes; only {available_bytes} available")
    submissions = plan_submissions(graph, operations,
                                   streamed=frozenset(value for value, binding in bindings.items()
                                                      if binding.residency == Residency.STREAMED))
    diagnostics = build_diagnostics(
        graph,
        operations,
        memory,
        submissions,
        compiler_identity=compiler_identity,
        configuration_identity=compiler_target.identity,
        mode=options.mode,
        precision=options.precision,
    )
    logger.info("Ops lowering: %s", diagnostics.render_summary())
    return CompilationPlan(graph, operations, memory, submissions, diagnostics,
                           bindings, execution_graph(graph, submissions, bindings),
                           device.fingerprint if device is not None else None, formula_graph)


def materialize(
    plan: CompilationPlan,
    *,
    device: DeviceRuntime,
    constants: Mapping[int | str, Resource | Binding] | None = None,
    static_resources: Mapping[int | str, Resource] | None = None,
) -> CompiledFunction:
    """Allocate and compile one previously analyzed plan."""
    graph, memory, submissions = plan.graph, plan.memory, plan.submissions
    if plan.device_fingerprint is not None and (
        device.configuration is None or device.configuration.fingerprint != plan.device_fingerprint
    ):
        raise ValueError("materialization runtime differs from the analyzed device plan")
    supplied = {**plan.bindings, **(constants or {})}
    streamed = {value: binding for value, binding in plan.bindings.items()
                if binding.residency == Residency.STREAMED}
    units = []
    imported: tuple[Resource, ...] = ()
    owned_storage: tuple[Resource, ...] = ()
    try:
        bound_constants, imported = _bind_constants(graph, supplied, device, streamed=frozenset(streamed))
        bound_resources = _bind_resources(graph, static_resources or {}, device)
        owned_storage, static_values, static_workspace = _allocate_temporary_slots(device, memory)
        for submission in submissions:
            compilation_unit = build_unit(graph, memory, submission)
            if len(submission.operations) == 1 and submission.operations[0].source_loop is not None:
                from ..runtime.streaming import compile_source_loop

                loop = compile_source_loop(device, submission.operations[0].source_loop)
                units.append(_CompiledUnit(compilation_unit, None, None, (), loop))
                continue
            if plan.execution is not None:
                for value in plan.execution.streamed_by_unit[submission.index]:
                    device.prepare_binding(streamed[value])
            static = {}
            for parameter in compilation_unit.parameters:
                if parameter.kind == ParameterKind.CONSTANT and parameter.key[1] not in streamed:
                    static[parameter.index] = bound_constants[parameter.key[1]].native
                elif parameter.kind == ParameterKind.TEMPORARY:
                    if parameter.key[0] == "value":
                        static[parameter.index] = static_values[parameter.key[1]].native
                    else:
                        candidate = next(
                            item.operation.name
                            for item in compilation_unit.calls
                            if min(item.operation.nodes) == parameter.key[1]
                        )
                        static[parameter.index] = static_workspace[
                            (candidate, parameter.key[2])
                        ].native
                elif (
                    parameter.kind == ParameterKind.RESOURCE and parameter.key[1] in bound_resources
                ):
                    static[parameter.index] = bound_resources[parameter.key[1]].native
            dynamic_indices = tuple(
                parameter.index
                for parameter in compilation_unit.parameters
                if (parameter.kind not in (ParameterKind.CONSTANT, ParameterKind.TEMPORARY)
                    or parameter.kind == ParameterKind.CONSTANT and parameter.key[1] in streamed)
                and not (
                    parameter.kind == ParameterKind.RESOURCE and parameter.key[1] in bound_resources
                )
            )
            executable = device.compile_native(compilation_unit, compilation_unit.signature)
            try:
                entrypoint = executable.bind(static, dynamic_indices)
            except BaseException:
                executable.close()
                raise
            dynamic = tuple(
                parameter.key
                for parameter in compilation_unit.parameters
                if (parameter.kind not in (ParameterKind.CONSTANT, ParameterKind.TEMPORARY)
                    or parameter.kind == ParameterKind.CONSTANT and parameter.key[1] in streamed)
                and not (
                    parameter.kind == ParameterKind.RESOURCE and parameter.key[1] in bound_resources
                )
            )
            units.append(_CompiledUnit(compilation_unit, executable, entrypoint, dynamic))
        compiled = CompiledFunction(
            device, graph, memory, tuple(units), bound_constants, bound_resources,
            owned_storage, static_values, static_workspace, plan.diagnostics,
            streamed, plan.execution, plan.formula_graph,
        )
    except BaseException:
        for unit in reversed(units):
            unit.close()
        for resource in reversed(owned_storage):
            resource.close()
        for resource in imported:
            resource.close()
        raise
    for resource in imported:
        resource.close()
    return compiled


def _allocate_temporary_slots(device: DeviceRuntime, memory: MemoryPlan):
    placements = [
        ("value", value_id, placement.slot, placement.spec)
        for value_id, placement in memory.values.items()
        if placement.storage == StorageClass.TEMPORARY
    ]
    placements.extend(
        ("workspace", index, placement.slot, placement.spec)
        for index, placement in enumerate(memory.workspace)
    )
    if not placements:
        return (), {}, {}
    # Every ABI tensor starts at byte offset zero. The memory plan has already
    # colored disjoint live intervals into reusable whole-allocation slots.
    slot_sizes: dict[int, int] = {}
    for _, _, slot, spec in placements:
        assert slot is not None
        slot_sizes[slot] = max(slot_sizes.get(slot, 0), spec.storage_nbytes)
    owned = []
    slots = {}
    values = {}
    workspaces = {}
    try:
        for slot_id, size in sorted(slot_sizes.items()):
            allocation = device.allocate_temporary(size, memory.alignment)
            slots[slot_id] = allocation
            owned.append(allocation)
        for kind, index, slot, spec in placements:
            assert slot is not None
            view = slots[slot].view(spec)
            if kind == "value":
                value_id = index
                values[value_id] = view
            else:
                placement = memory.workspace[index]
                workspaces[(placement.operation, placement.index)] = view
            owned.append(view)
        return tuple(owned), values, workspaces
    except BaseException:
        for resource in reversed(owned):
            resource.close()
        raise


def _bind_constants(
    graph: Graph, supplied: Mapping[int | str, Resource | Binding], device: DeviceRuntime,
    *, streamed: frozenset[int] = frozenset(),
) -> tuple[dict[int, Resource], tuple[Resource, ...]]:
    result = {}
    imported = []
    try:
        for value_id in graph.constants:
            value = graph.values[value_id]
            resource = supplied.get(value_id)
            if resource is None and value.name is not None:
                resource = supplied.get(value.name)
            if resource is None:
                raise KeyError(f"missing immutable constant {value.name or value_id}")
            if value_id in streamed:
                if not isinstance(resource, Binding):
                    raise ValueError("a streamed source cannot be replaced after physical planning")
                continue
            if isinstance(resource, Binding):
                resource = device.resolve(resource)
                imported.append(resource)
            _check_binding(device, value.spec, resource)
            result[value_id] = resource
        return result, tuple(imported)
    except BaseException:
        for resource in imported:
            resource.close()
        raise


def _bind_resources(
    graph: Graph, supplied: Mapping[int | str, Resource], device: DeviceRuntime
) -> dict[int, Resource]:
    result = {}
    for value_id in graph.resources:
        value = graph.values[value_id]
        resource = supplied.get(value_id)
        if resource is None and value.name is not None:
            resource = supplied.get(value.name)
        if resource is None:
            continue
        _check_binding(device, value.spec, resource)
        result[value_id] = resource
    unknown_names = {
        key
        for key in supplied
        if isinstance(key, str)
        and key not in {graph.values[value_id].name for value_id in graph.resources}
    }
    if unknown_names:
        raise KeyError(f"unknown static resources: {sorted(unknown_names)}")
    return result


def _specialize_signature(signature: Signature, dimensions: Mapping[str, int]) -> Signature:
    from ..tensor.tracing import Argument
    from ..tensor.tree import map_tree

    def bind(argument, path):
        return replace(argument, spec=argument.spec.bind(dict(dimensions))) if isinstance(argument, Argument) else argument

    try:
        return Signature(
            tuple(map_tree(argument, bind) for argument in signature.args),
            {name: map_tree(argument, bind) for name, argument in signature.kwargs.items()},
            signature.static_kwargs,
        )
    except KeyError as error:
        raise ValueError(
            f"missing specialization for symbolic dimension {error.args[0]!r}"
        ) from error


def _check_binding(device: DeviceRuntime, expected: TensorSpec, resource: Resource) -> None:
    if resource.device is not device:
        raise ValueError("resource belongs to another device")
    if resource.spec != expected:
        raise ValueError(f"resource specification {resource.spec!r} differs from {expected!r}")
