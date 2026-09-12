"""Compile-free Qwen lowering qualification against real artifact metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import magnitensor as mt
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
class QualificationResult:
    case: QualificationCase
    graph_fingerprint: str
    nodes: int
    kernels: int
    submissions: int
    temporary_bytes: int
    selected: dict[str, int]
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
        recurrent_offsets=(
            mt.TensorSpec((case.batch + 1,), mt.DType.I32) if recurrent else None
        ),
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
        item.name.split("@", 1)[0]
        for item in plan.diagnostics.candidates
        if item.selected
    )
    failures = _failures(description, case, selected, len(plan.submissions))
    return QualificationResult(
        case,
        plan.graph.fingerprint,
        len(plan.graph.nodes),
        plan.diagnostics.dispatches,
        len(plan.submissions),
        plan.memory.temporary_bytes,
        dict(sorted(selected.items())),
        tuple(failures),
    )


def _failures(
    description: DenseDescription,
    case: QualificationCase,
    selected: Counter[str],
    submissions: int,
) -> list[str]:
    failures = []
    for operation in ("linear.portable", "matmul.portable"):
        if selected[operation]:
            failures.append(f"selected forbidden scalar contraction {operation}")
    if submissions != 1:
        failures.append(f"expected one maximal submission, selected {submissions}")
    attention = sum(isinstance(block.mixer, AttentionWeights) for block in description.blocks)
    recurrent = len(description.blocks) - attention
    selected_attention = sum(
        count for name, count in selected.items() if name.startswith("causal_attention.")
    )
    if case.logits and selected_attention != attention:
        failures.append("not every attention layer selected an attention schedule")
    if case.logits and selected["recurrent_prepare.channel-parallel"] != recurrent:
        failures.append("not every recurrent layer selected channel-parallel preparation")
    if case.name == "decode-b1":
        expected = len(description.blocks)
        actual = selected["linear.parallel-direct"]
        if actual != expected:
            failures.append(
                f"expected {expected} fused parallel projection regions, selected {actual}"
            )
    routed = sum(
        isinstance(block.feedforward, RoutedFeedForwardWeights)
        for block in description.blocks
    )
    dense = sum(
        isinstance(block.feedforward, DenseFeedForwardWeights)
        for block in description.blocks
    )
    if routed and case.logits:
        expected = "routed_experts.grouped" if case.mode == "prefill" else "routed_experts.direct"
        if selected[expected] != routed:
            failures.append(f"expected {routed} {expected} regions, selected {selected[expected]}")
        if selected["routed_experts.portable"]:
            failures.append("selected portable routed experts")
    if dense and case.logits and case.mode == "prefill" and case.rows > 4:
        if selected["dense_swiglu.matrix"] != dense:
            actual = selected["dense_swiglu.matrix"]
            failures.append(
                f"expected {dense} matrix SwiGLU regions, selected {actual}"
            )
    return failures


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
        print(json.dumps([asdict(result) for result in results], indent=2))
        if any(result.failures for result in results):
            raise SystemExit(1)
    finally:
        residency.close()
        device.close()
        format.close()


if __name__ == "__main__":
    main()
