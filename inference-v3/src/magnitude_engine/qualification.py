"""Compile-free Qwen lowering qualification against real artifact metadata."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import magnitensor as mt
from magnitensor.compiler.unit import ParameterKind, build_unit
from magnitensor.kernels.packed import packet_format
from magnitude_engine.models.qwen35.description import (
    AttentionWeights,
    DenseDescription,
    DenseFeedForwardWeights,
    MixerKind,
    RoutedFeedForwardWeights,
)
from magnitude_engine.models.qwen35.tensor_program import (
    InvocationSpecs,
    define,
    weight_roles,
)
from magnitude_engine.weights.formats.gguf import GGUFFormat
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat
from magnitude_engine.weights.tensor_residency import TensorWeights


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
    capabilities: dict[str, Any]
    graph_fingerprint: str
    nodes: int
    kernels: int
    submissions: int
    temporary_bytes: int
    bound_parameters: int
    dynamic_parameters: int
    largest_region_nodes: int
    representation_bytes: dict[str, int]
    selected: dict[str, int]
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
    tokens = mt.TensorSpec((case.rows,), mt.DType.I32)
    destinations = mt.TensorSpec((case.rows,), mt.DType.I32)
    visible = mt.TensorSpec((case.rows, 2), mt.DType.I32)
    cache = mt.TensorSpec(
        (2, slots * case.context, g.kv_heads, g.attention_width),
        g.activation_dtype,
    )
    convolution = mt.TensorSpec(
        (1, g.recurrent_channels, g.convolution_width - 1), g.activation_dtype
    )
    delta = mt.TensorSpec(
        (1, g.recurrent_value_heads, g.recurrent_width, g.recurrent_width), mt.DType.F32
    )
    output_rows = mt.TensorSpec((case.batch,), mt.DType.I32) if case.logits else None
    draws = mt.TensorSpec((case.batch, 6), mt.DType.U32) if case.logits else None
    return InvocationSpecs(
        batch=case.batch,
        tokens=tokens,
        coordinates=mt.TensorSpec((case.rows, 3), mt.DType.I32),
        recurrent_offsets=(mt.TensorSpec((case.batch + 1,), mt.DType.I32) if recurrent else None),
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
    weight_specs: dict[str, mt.TensorSpec],
    capabilities: mt.Capabilities,
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
    plan = mt.analyze(
        definition.function,
        signature=definition.signature,
        capabilities=capabilities,
        compiler_identity=compiler_identity,
        available_bytes=available_bytes,
        options=definition.options,
    )
    selected = Counter(
        item.name.split("@", 1)[0] for item in plan.diagnostics.candidates if item.selected
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
        _schedule_manifest(plan.graph, candidate) for candidate in plan.cover.candidates
    )
    representation_bytes = Counter()
    for spec in weight_specs.values():
        representation_bytes[_representation_name(spec)] += spec.storage_nbytes
    failures = _failures(
        description,
        case,
        selected,
        len(plan.submissions),
        weight_specs,
        plan.cover.candidates,
        units,
    )
    return QualificationResult(
        case,
        _capability_manifest(capabilities),
        plan.graph.fingerprint,
        len(plan.graph.nodes),
        plan.diagnostics.dispatches,
        len(plan.submissions),
        plan.memory.temporary_bytes,
        bound_parameters,
        dynamic_parameters,
        max((len(candidate.nodes) for candidate in plan.cover.candidates), default=0),
        dict(sorted(representation_bytes.items())),
        dict(sorted(selected.items())),
        schedules,
        programs,
        tuple(failures),
    )


def _schedule_manifest(graph, candidate: mt.Candidate) -> ScheduleManifest:
    emitter = candidate.emitter
    geometry = {
        name: _manifest_value(value)
        for name, value in sorted(vars(emitter).items())
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


def _capability_manifest(capabilities: mt.Capabilities) -> dict[str, Any]:
    return {
        "subgroup_width": capabilities.subgroup_width,
        "threads_per_group": capabilities.threads_per_group,
        "shared_memory_bytes": capabilities.shared_memory_bytes,
        "matrix_instructions": tuple(
            f"{item.input_dtype.value}:{item.m}x{item.n}x{item.k}->{item.accumulation_dtype.value}"
            for item in capabilities.matrix_instructions
        ),
        "memory_scopes": tuple(sorted(capabilities.memory_scopes)),
        "atomics": tuple(sorted(dtype.value for dtype in capabilities.atomics)),
        "native_multi_launch": capabilities.native_multi_launch,
        "partial_binding": capabilities.partial_binding,
        "fingerprint": capabilities.fingerprint,
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
        sum(call.candidate.kernel_count for call in unit.calls),
        tuple(call.candidate.name for call in unit.calls),
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


def _spec_manifest(spec: mt.TensorSpec) -> str:
    shape = "x".join(str(value) for value in spec.shape)
    return f"{shape}:{spec.dtype.value}:{_representation_name(spec)}"


def _manifest_value(value: Any) -> Any:
    if isinstance(value, mt.TensorSpec):
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


def _failures(
    description: DenseDescription,
    case: QualificationCase,
    selected: Counter[str],
    submissions: int,
    weight_specs: dict[str, mt.TensorSpec],
    candidates: tuple[mt.Candidate, ...],
    units,
) -> list[str]:
    failures = []
    forbidden_primitives = {
        "linear.portable",
        "matmul.portable",
        "embedding.portable",
        "route_topk.portable",
        "routed_experts.portable",
        "causal_attention.portable",
        "attention_prepare.portable",
        "kv_append.portable",
        "recurrent_prepare.portable",
        "gated_delta_recurrence.portable",
        "rms_norm.portable",
        "row_dot.portable",
        "reshape.portable",
    }
    for operation in sorted(forbidden_primitives):
        if selected[operation]:
            failures.append(f"selected forbidden production fallback {operation}")
    for name in selected:
        if any(
            marker in name
            for marker in (
                "direct-encoded",
                "fragment-encoded",
                "scalar",
                "causal_attention.online",
            )
        ):
            failures.append(f"selected obsolete implementation {name}")
    expected_region_kernels = {
        "routed_experts.grouped": 4,
        "routed_experts.packet-shared": 2,
        "dense_swiglu.packet-prefill": 2,
        "dense_swiglu.packet-decode": 2,
        "attention.matrix-streaming-gated-output": 2,
        "attention.register-partitioned-gated-output": 3,
        "recurrent.output-prefill": 2,
        "recurrent.output-decode": 2,
        "gated_delta.chunked-matrix": 2,
    }
    for candidate in candidates:
        family = candidate.name.split("@", 1)[0]
        expected_kernels = expected_region_kernels.get(family)
        if family == "attention.matrix-streaming-gated-output":
            expected_kernels = 2 + (getattr(candidate.emitter, "partitions", 1) > 1)
        if expected_kernels is not None and candidate.kernel_count != expected_kernels:
            failures.append(
                f"{family} uses {candidate.kernel_count} kernels; expected {expected_kernels}"
            )
    if submissions != 1:
        failures.append(f"expected one maximal submission, selected {submissions}")
    if any(len(unit.calls) != len({call.candidate.name for call in unit.calls}) for unit in units):
        failures.append("a maximal program contains duplicate candidate identities")
    for name, spec in weight_specs.items():
        representation = spec.representation
        if (
            isinstance(representation, mt.Affine)
            and representation.code.low_bits == 4
            and representation.group == 32
            and isinstance(representation.coefficients, mt.DirectCoefficients)
            and representation.coefficients.scale_dtype == mt.DType.F32
        ):
            failures.append(f"hierarchical quantization was expanded for {name}")
    attention = sum(isinstance(block.mixer, AttentionWeights) for block in description.blocks)
    recurrent = len(description.blocks) - attention
    if selected["attention.prepare-append"] != attention:
        failures.append(
            f"expected {attention} fused attention prepare/append regions, "
            f"selected {selected['attention.prepare-append']}"
        )
    if selected["recurrent_prepare.channel-parallel"] != recurrent:
        failures.append("not every recurrent layer selected channel-parallel preparation")
    recurrence_family = (
        "gated_delta.chunked-matrix"
        if case.mode == "prefill" and case.rows >= 64 * case.batch
        else "gated_delta.register-state"
    )
    if selected[recurrence_family] != recurrent:
        failures.append(f"not every recurrent layer selected {recurrence_family}")
    if (
        case.mode == "prefill"
        and case.logits
        and selected["attention.matrix-streaming-gated-output"] != attention
    ):
        failures.append(
            "not every attention layer selected a complete matrix attention/gate/output pipeline"
        )
    if case.name == "decode-b1":
        expected = len(description.blocks)
        actual = selected["linear.parallel-packet-decode"]
        if actual != expected:
            failures.append(
                f"expected {expected} fused parallel projection regions, selected {actual}"
            )
    elif case.mode == "prefill":
        expected = len(description.blocks)
        actual = selected["linear.parallel-packet-prefill"]
        if actual != expected:
            failures.append(
                f"expected {expected} tiled parallel projection regions, selected {actual}"
            )
    routed = sum(
        isinstance(block.feedforward, RoutedFeedForwardWeights) for block in description.blocks
    )
    dense = sum(
        isinstance(block.feedforward, DenseFeedForwardWeights) for block in description.blocks
    )
    if routed and case.logits:
        if case.name == "decode-b1":
            if selected["route_topk.fused-router"] != routed:
                failures.append(
                    f"expected {routed} fused router/top-k regions, "
                    f"selected {selected['route_topk.fused-router']}"
                )
        elif selected["route_topk.subgroup"] != routed:
            failures.append(
                f"expected {routed} subgroup top-k regions after matrix router projection, "
                f"selected {selected['route_topk.subgroup']}"
            )
        expected = (
            "routed_experts.grouped" if case.mode == "prefill" else "routed_experts.packet-shared"
        )
        if selected[expected] != routed:
            failures.append(f"expected {routed} {expected} regions, selected {selected[expected]}")
    if case.logits:
        expected_dense = dense
        dense_name = f"dense_swiglu.packet-{case.mode}"
        if selected[dense_name] != expected_dense:
            failures.append(
                f"expected {expected_dense} {dense_name} regions, selected {selected[dense_name]}"
            )
    if case.mode == "decode" and case.logits:
        if selected["attention.register-partitioned-gated-output"] != attention:
            failures.append("not every attention layer fused partition merge, gate, and output")
        if selected["recurrent.output-decode"] != recurrent:
            failures.append("not every recurrent layer fused normalization, gate, and output")
    failures.extend(_inconsistent_schedule_failures(candidates))
    failures.extend(_source_policy_failures())
    return failures


def _inconsistent_schedule_failures(candidates: tuple[mt.Candidate, ...]) -> list[str]:
    observed: dict[tuple[str, tuple[str, ...], tuple[str, ...]], str] = {}
    failures = []
    for candidate in candidates:
        family = candidate.name.split("@", 1)[0]
        emitter_specs = getattr(candidate.emitter, "specs", None)
        output_specs = getattr(candidate.emitter, "output_specs", None)
        if emitter_specs is None and output_specs is None:
            continue
        inputs = tuple(repr(spec) for spec in emitter_specs or ())
        outputs = tuple(repr(spec) for spec in output_specs or ())
        key = family, inputs, outputs
        geometry = repr(
            {
                name: value
                for name, value in vars(candidate.emitter).items()
                if name not in {"specs", "weight_specs", "output_specs"}
            }
        )
        previous = observed.setdefault(key, geometry)
        if previous != geometry:
            failures.append(f"identical {family} shapes selected different schedule geometry")
    return failures


def _source_policy_failures() -> list[str]:
    root = Path(mt.__file__).resolve().parent
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


def _representation_name(spec: mt.TensorSpec) -> str:
    representation = spec.representation
    if representation is None or isinstance(representation, mt.Dense):
        return f"dense-{spec.dtype.value}"
    if isinstance(representation, mt.Affine):
        coefficients = representation.coefficients
        family = (
            "hierarchical" if isinstance(coefficients, mt.HierarchicalCoefficients) else "direct"
        )
        return f"affine-{representation.code.bits}bit-g{representation.group}-{family}"
    if isinstance(representation, mt.Codebook):
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
    runtime = mt.TileLangRuntime(args.backend)
    device = mt.Device(runtime, budget_bytes=args.memory_bytes)
    residency = TensorWeights(format, device)
    try:
        if path.is_dir():
            from magnitude_engine.models.qwen35.formats.mlx import describe
        else:
            from magnitude_engine.models.qwen35.formats.gguf import describe

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
                device.capabilities,
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
