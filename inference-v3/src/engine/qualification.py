"""Compile-free Qwen lowering qualification against real artifact metadata."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import ops
from ops.compiler.unit import ParameterKind, build_unit
from ops.compiler.lowering import BoundOperation
from engine import DevicePlan
from ops.kernels.packed import packet_format
from engine.models.qwen35.description import (
    AttentionWeights,
    DenseDescription,
    DenseFeedForwardWeights,
    MixerKind,
    RoutedFeedForwardWeights,
)
from engine.models.qwen35.tensor_program import (
    InvocationSpecs,
    define,
    weight_roles,
)
from engine.weights.formats.gguf import GGUFFormat
from engine.weights.formats.mlx_safetensors import MLXFormat
from engine.weights.tensor_residency import TensorWeights


@dataclass(frozen=True, slots=True)
class QualificationCase:
    name: str
    mode: str
    rows: int
    batch: int
    context: int
    logits: bool


@dataclass(frozen=True, slots=True)
class ScheduleManifest:
    name: str
    nodes: tuple[int, ...]
    kernel_count: int
    workspace_bytes: int
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    packet_formats: tuple[str, ...]
    packet_geometry: dict[str, dict[str, int]]
    geometry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParameterManifest:
    name: str
    kind: str
    binding: str
    specification: str


@dataclass(frozen=True, slots=True)
class ProgramManifest:
    name: str
    reason: str
    kernel_count: int
    schedules: tuple[str, ...]
    parameters: tuple[ParameterManifest, ...]


@dataclass(frozen=True, slots=True)
class QualificationResult:
    case: QualificationCase
    compiler_target: dict[str, Any]
    graph_fingerprint: str
    nodes: int
    kernels: int
    submissions: int
    temporary_bytes: int
    bound_parameters: int
    dynamic_parameters: int
    largest_region_nodes: int
    representation_bytes: dict[str, int]
    operations: dict[str, int]
    schedules: tuple[ScheduleManifest, ...]
    programs: tuple[ProgramManifest, ...]
    failures: tuple[str, ...]


def invocation_specs(
    description: DenseDescription,
    case: QualificationCase,
    *,
    slots: int,
) -> InvocationSpecs:
    g = description.geometry
    attention = sum(kind == MixerKind.ATTENTION for kind in g.layers)
    recurrent = len(g.layers) - attention
    tokens = ops.TensorSpec((case.rows,), ops.DType.I32)
    destinations = ops.TensorSpec((case.rows,), ops.DType.I32)
    visible = ops.TensorSpec((case.rows, 2), ops.DType.I32)
    cache = ops.TensorSpec(
        (2, slots * case.context, g.kv_heads, g.attention_width),
        g.activation_dtype,
    )
    convolution = ops.TensorSpec(
        (1, g.recurrent_channels, g.convolution_width - 1), g.activation_dtype
    )
    delta = ops.TensorSpec(
        (1, g.recurrent_value_heads, g.recurrent_width, g.recurrent_width), ops.DType.F32
    )
    output_rows = ops.TensorSpec((case.batch,), ops.DType.I32) if case.logits else None
    draws = ops.TensorSpec((case.batch, 6), ops.DType.U32) if case.logits else None
    return InvocationSpecs(
        batch=case.batch,
        tokens=tokens,
        coordinates=ops.TensorSpec((case.rows, 3), ops.DType.I32),
        recurrent_offsets=(ops.TensorSpec((case.batch + 1,), ops.DType.I32) if recurrent else None),
        output_rows=output_rows,
        draws=draws,
        destinations=tuple(destinations for _ in range(attention)),
        visible=tuple(visible for _ in range(attention)),
        attention_state=tuple(cache for _ in range(attention)),
        convolution_state=tuple(convolution for _ in range(recurrent * case.batch)),
        delta_state=tuple(delta for _ in range(recurrent * case.batch)),
    )


def qualify(
    description: DenseDescription,
    weight_specs: dict[str, ops.TensorSpec],
    compiler_target: ops.CompilerTarget,
    compiler_identity: str,
    case: QualificationCase,
    *,
    slots: int,
    available_bytes: int,
) -> QualificationResult:
    definition = define(
        description,
        weight_specs,
        case.mode,
        invocation_specs(description, case, slots=slots),
    )
    plan = ops.analyze(
        definition.function,
        signature=definition.signature,
        compiler_target=compiler_target,
        compiler_identity=compiler_identity,
        available_bytes=available_bytes,
        options=definition.options,
    )
    selected = Counter(
        item.name.split("@", 1)[0] for item in plan.diagnostics.operations
    )
    units = tuple(build_unit(plan.graph, plan.memory, unit) for unit in plan.submissions)
    programs = tuple(
        _program_manifest(unit, submission.reason)
        for unit, submission in zip(units, plan.submissions, strict=True)
    )
    parameters = tuple(parameter for program in programs for parameter in program.parameters)
    bound_parameters = sum(parameter.binding == "bound" for parameter in parameters)
    dynamic_parameters = sum(parameter.binding == "dynamic" for parameter in parameters)
    schedules = tuple(
        _schedule_manifest(plan.graph, candidate) for candidate in plan.operations
    )
    representation_bytes = Counter()
    for spec in weight_specs.values():
        representation_bytes[_representation_name(spec)] += spec.storage_nbytes
    failures = _failures(plan, units)
    return QualificationResult(
        case,
        _target_manifest(compiler_target),
        plan.graph.fingerprint,
        len(plan.graph.nodes),
        plan.diagnostics.dispatches,
        len(plan.submissions),
        plan.memory.temporary_bytes,
        bound_parameters,
        dynamic_parameters,
        max((len(candidate.nodes) for candidate in plan.operations), default=0),
        dict(sorted(representation_bytes.items())),
        dict(sorted(selected.items())),
        schedules,
        programs,
        tuple(failures),
    )


def _schedule_manifest(graph, candidate: BoundOperation) -> ScheduleManifest:
    emitter = candidate.emitter
    geometry = {
        name: _manifest_value(value)
        for name, value in sorted(_emitter_fields(emitter).items())
        if name not in {"specs", "weight_specs", "output_specs"}
    }
    packets = {
        packet.name: {
            "dot_packet": packet.dot_packet,
            "matrix_packet": packet.matrix_packet,
            "reduction_tile": packet.tile,
        }
        for value in (*candidate.inputs, *candidate.outputs)
        if (packet := packet_format(graph.values[value].spec)) is not None
    }
    return ScheduleManifest(
        candidate.name,
        tuple(sorted(candidate.nodes)),
        candidate.kernel_count,
        candidate.workspace_bytes,
        tuple(_value_manifest(graph.values[value]) for value in candidate.inputs),
        tuple(_value_manifest(graph.values[value]) for value in candidate.outputs),
        tuple(sorted(packets)),
        dict(sorted(packets.items())),
        geometry,
    )


def _target_manifest(compiler_target: ops.CompilerTarget) -> dict[str, Any]:
    return {
        "subgroup_width": compiler_target.subgroup_width,
        "threads_per_group": compiler_target.threads_per_group,
        "shared_memory_bytes": compiler_target.shared_memory_bytes,
        "fingerprint": compiler_target.identity,
    }


def _program_manifest(unit, reason: str) -> ProgramManifest:
    parameters = tuple(
        ParameterManifest(
            parameter.name,
            parameter.kind.value,
            "bound" if _statically_bound(parameter) else "dynamic",
            _spec_manifest(parameter.spec),
        )
        for parameter in unit.parameters
    )
    return ProgramManifest(
        unit.name,
        reason,
        sum(call.operation.kernel_count for call in unit.calls),
        tuple(call.operation.name for call in unit.calls),
        parameters,
    )


def _statically_bound(parameter) -> bool:
    if parameter.kind in (ParameterKind.CONSTANT, ParameterKind.TEMPORARY):
        return True
    return (
        parameter.kind == ParameterKind.RESOURCE
        and parameter.name.startswith("attention.")
        and parameter.name.endswith(".state")
    )


def _value_manifest(value) -> str:
    return f"{value.name or 'v' + str(value.id)}:{_spec_manifest(value.spec)}"


def _spec_manifest(spec: ops.TensorSpec) -> str:
    shape = "x".join(str(value) for value in spec.shape)
    return f"{shape}:{spec.dtype.value}:{_representation_name(spec)}"


def _manifest_value(value: Any) -> Any:
    if isinstance(value, ops.TensorSpec):
        return _spec_manifest(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _manifest_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_manifest_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _emitter_fields(emitter):
    return ({field.name: getattr(emitter, field.name) for field in fields(emitter)}
            if is_dataclass(emitter) else vars(emitter))


def _failures(plan: ops.CompilationPlan, units) -> list[str]:
    """Structural qualification, independent of a particular kernel decomposition.

    Numerical checks and formula measurements belong to the consolidated gate.
    A kernel count or emitter family name is diagnostic data, not a speed proof.
    """
    failures = []
    owned = tuple(node for operation in plan.operations for node in operation.nodes)
    if len(owned) != len(set(owned)) or set(owned) != set(range(len(plan.graph.nodes))):
        failures.append("physical operations do not own mathematical work exactly once")
    for operation in plan.operations:
        if operation.kernel_count and operation.definition is None and operation.source_loop is None:
            failures.append(f"{operation.name} has no authored executable definition")
    dispatched = tuple(id(call.operation) for unit in units for call in unit.calls)
    expected = tuple(id(operation) for operation in plan.operations
                     if operation.kernel_count and operation.source_loop is None)
    if Counter(dispatched) != Counter(expected):
        failures.append("submission programs do not dispatch their defined operations exactly once")
    failures.extend(_source_policy_failures())
    return failures


def _source_policy_failures() -> list[str]:
    root = Path(ops.__file__).resolve().parent
    failures = []
    forbidden_imports = (
        "tilelang.metal",
        "tilelang.cuda",
        "tilelang.hip",
        "mlx",
        "numpy",
        "torch",
    )
    backend_names = {"metal", "cuda", "hip", "rocm"}
    for path in sorted((root / "kernels").glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = () if node.module is None else (node.module,)
            else:
                names = ()
            for name in names:
                if any(name == item or name.startswith(item + ".") for item in forbidden_imports):
                    failures.append(f"forbidden production import {name} in {path.name}")
            if isinstance(node, ast.Constant) and node.value in backend_names:
                failures.append(f"backend-name branch marker {node.value!r} in {path.name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"call_extern", "call_pure_extern", "call_intrin"}:
                    failures.append(f"forbidden backend intrinsic call in {path.name}")
    return sorted(set(failures))


def _representation_name(spec: ops.TensorSpec) -> str:
    representation = spec.representation
    if representation is None or isinstance(representation, ops.Dense):
        return f"dense-{spec.dtype.value}"
    if isinstance(representation, ops.Affine):
        coefficients = representation.coefficients
        family = (
            "hierarchical" if isinstance(coefficients, ops.HierarchicalCoefficients) else "direct"
        )
        return f"affine-{representation.code.bits}bit-g{representation.group}-{family}"
    if isinstance(representation, ops.Codebook):
        return f"codebook-{representation.code_bits}bit-g{representation.group}"
    raise TypeError(f"unknown representation {representation!r}")


def standard_cases(contexts: tuple[int, int], max_batch: int) -> tuple[QualificationCase, ...]:
    short, long = contexts
    return (
        QualificationCase("prefill-128-state", "prefill", 128, 1, short, False),
        QualificationCase("prefill-512-logits", "prefill", 512, 1, short, True),
        QualificationCase("prefill-2048-logits", "prefill", 2048, 1, long, True),
        QualificationCase("decode-b1", "decode", 1, 1, long, True),
        QualificationCase("decode-max-batch", "decode", max_batch, max_batch, short, True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--backend", default="metal", choices=("metal", "cuda", "hip", "llvm"))
    parser.add_argument("--contexts", default="16384,65536")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--memory-bytes", type=int, default=128 * 1024**3)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the complete deterministic JSON manifest to this path",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="run only this named standard case; repeat to select several",
    )
    args = parser.parse_args()
    contexts = tuple(int(value) for value in args.contexts.split(","))
    if len(contexts) != 2 or any(value <= 0 for value in contexts):
        parser.error("--contexts requires two positive comma-separated capacities")
    path = Path(args.target).expanduser().resolve(strict=True)
    format = MLXFormat(str(path)) if path.is_dir() else GGUFFormat(str(path))
    device = ops.DeviceRuntime.open(DevicePlan.discover(
        backend=args.backend, maximum_bytes=args.memory_bytes,
    ))
    residency = TensorWeights(format, device)
    try:
        if path.is_dir():
            from engine.models.qwen35.formats.mlx import describe
        else:
            from engine.models.qwen35.formats.gguf import describe

        description = describe(format)  # type: ignore[arg-type]
        weight_specs = {
            descriptor.name: residency.spec(descriptor, dtype)
            for descriptor, dtype in weight_roles(description)
        }
        cases = standard_cases(contexts, args.max_batch)  # type: ignore[arg-type]
        if args.cases:
            requested = set(args.cases)
            unknown = requested - {case.name for case in cases}
            if unknown:
                parser.error(f"unknown qualification cases: {sorted(unknown)}")
            cases = tuple(case for case in cases if case.name in requested)
        results = tuple(
            qualify(
                description,
                weight_specs,
                device.compiler_target,
                device.compiler_identity,
                case,
                slots=args.max_batch,
                available_bytes=device.available_bytes,
            )
            for case in cases
        )
        rendered = json.dumps([asdict(result) for result in results], indent=2) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            output = args.output.expanduser().resolve()
            output.write_text(rendered)
            print(
                f"wrote {output}: {len(results)} cases, "
                f"{sum(len(result.failures) for result in results)} failures"
            )
        if any(result.failures for result in results):
            raise SystemExit(1)
    finally:
        residency.close()
        device.close()
        format.close()


if __name__ == "__main__":
    main()
