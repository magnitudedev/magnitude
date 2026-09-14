"""Focused Magnitensor schedule benchmarks without model loading or serving."""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import magnitensor as mt


@dataclass(frozen=True, slots=True)
class Operand:
    argument: mt.Argument
    content: bytes | None = None


@dataclass(frozen=True, slots=True)
class KernelCase:
    function: Callable[..., object]
    operands: tuple[Operand, ...]


@dataclass(frozen=True, slots=True)
class KernelResult:
    case: str
    selected: tuple[str, ...]
    kernels: int
    warmup: int
    repeat: int
    median_ms: float
    minimum_ms: float
    maximum_ms: float


def _encoded(shape: tuple[int, ...]) -> mt.TensorSpec:
    return mt.TensorSpec(shape, mt.DType.F16).with_representation(
        mt.Affine(
            mt.Code(4),
            64,
            mt.DirectCoefficients(mt.DType.BF16, mt.DType.BF16),
        )
    )


def _linear(rows: int, width: int, output: int) -> KernelCase:
    hidden = mt.TensorSpec((rows, width), mt.DType.F16)
    weight = _encoded((output, width))
    return KernelCase(
        lambda value, projection: mt.linear(value, projection),
        (
            Operand(mt.Argument(hidden, "hidden")),
            Operand(mt.Argument(weight, "projection", mt.ValueKind.CONSTANT)),
        ),
    )


def _parallel_linear(rows: int, width: int, output: int) -> KernelCase:
    hidden = mt.TensorSpec((rows, width), mt.DType.F16)
    wide = _encoded((output, width))
    narrow = _encoded((max(8, output // 8), width))
    return KernelCase(
        lambda value, query, key, projected_value: (
            mt.linear(value, query),
            mt.linear(value, key),
            mt.linear(value, projected_value),
        ),
        (
            Operand(mt.Argument(hidden, "hidden")),
            Operand(mt.Argument(wide, "query", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(narrow, "key", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(narrow, "value", mt.ValueKind.CONSTANT)),
        ),
    )


def _dense_swiglu(rows: int, width: int, intermediate: int) -> KernelCase:
    hidden = mt.TensorSpec((rows, width), mt.DType.F16)
    gate = _encoded((intermediate, width))
    down = _encoded((width, intermediate))

    def function(value, gate_weight, up_weight, down_weight):
        activated = mt.silu(mt.linear(value, gate_weight)) * mt.linear(value, up_weight)
        return mt.linear(activated, down_weight)

    return KernelCase(
        function,
        (
            Operand(mt.Argument(hidden, "hidden")),
            Operand(mt.Argument(gate, "gate", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(gate, "up", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(down, "down", mt.ValueKind.CONSTANT)),
        ),
    )


def _attention(rows: int, context: int) -> KernelCase:
    query = mt.TensorSpec((rows, 16, 256), mt.DType.F16)
    history = mt.TensorSpec((2, context, 4, 256), mt.DType.F16)
    visible = mt.TensorSpec((rows, 2), mt.DType.I32)
    limits = tuple(value for _ in range(rows) for value in (0, context))
    return KernelCase(
        lambda value, limits, cache: mt.causal_attention(value, cache, limits, sequence_count=1),
        (
            Operand(mt.Argument(query, "query")),
            Operand(
                mt.Argument(visible, "visible"),
                struct.pack(f"={len(limits)}i", *limits),
            ),
            Operand(mt.Argument(history, "history", mt.ValueKind.RESOURCE)),
        ),
    )


def _recurrent_prepare(rows: int) -> KernelCase:
    key_heads, value_heads, width, history = 16, 32, 128, 3
    channels = (2 * key_heads + value_heads) * width
    projected = mt.TensorSpec((rows, channels), mt.DType.F16)
    convolution = mt.TensorSpec((channels, history + 1), mt.DType.F32)
    previous = mt.TensorSpec((1, channels, history), mt.DType.F16)
    parameter = mt.TensorSpec((rows, value_heads), mt.DType.F16)
    head_parameter = mt.TensorSpec((value_heads,), mt.DType.F32)
    offsets = struct.pack("=ii", 0, rows)

    def function(value, conv, state, alpha, beta, rate, bias, ranges):
        return mt.recurrent_prepare(
            value,
            conv,
            state,
            alpha,
            beta,
            rate,
            bias,
            ranges,
            key_heads=key_heads,
            value_heads=value_heads,
            width=width,
            convolution_width=history + 1,
            epsilon=1e-6,
        )

    return KernelCase(
        function,
        (
            Operand(mt.Argument(projected, "projected")),
            Operand(mt.Argument(convolution, "convolution", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(previous, "previous", mt.ValueKind.RESOURCE)),
            Operand(mt.Argument(parameter, "alpha")),
            Operand(mt.Argument(parameter, "beta")),
            Operand(mt.Argument(head_parameter, "rate", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(head_parameter, "bias", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(mt.TensorSpec((2,), mt.DType.I32), "offsets"), offsets),
        ),
    )


def _gated_recurrence(rows: int) -> KernelCase:
    key_heads, value_heads, width = 16, 32, 128
    key = mt.TensorSpec((rows, key_heads, width), mt.DType.F16)
    value = mt.TensorSpec((rows, value_heads, width), mt.DType.F16)
    parameter = mt.TensorSpec((rows, value_heads), mt.DType.F32)
    state = mt.TensorSpec((1, value_heads, width, width), mt.DType.F32)
    offsets = struct.pack("=ii", 0, rows)
    return KernelCase(
        lambda query, keys, values, decay, beta, previous, ranges: mt.gated_delta_recurrence(
            query,
            keys,
            values,
            decay,
            beta,
            previous,
            ranges,
            mapping="tiled",
        ),
        (
            Operand(mt.Argument(key, "query")),
            Operand(mt.Argument(key, "key")),
            Operand(mt.Argument(value, "value")),
            Operand(mt.Argument(parameter, "decay")),
            Operand(mt.Argument(mt.TensorSpec((rows, value_heads), mt.DType.F16), "beta")),
            Operand(mt.Argument(state, "previous", mt.ValueKind.RESOURCE)),
            Operand(mt.Argument(mt.TensorSpec((2,), mt.DType.I32), "offsets"), offsets),
        ),
    )


def _experts(rows: int) -> KernelCase:
    width, intermediate, experts, selected = 512, 512, 32, 4
    hidden = mt.TensorSpec((rows, width), mt.DType.F16)
    routes = mt.TensorSpec((rows, selected), mt.DType.I32)
    scores = mt.TensorSpec((rows, selected), mt.DType.F32)
    gate = _encoded((experts, intermediate, width))
    down = _encoded((experts, width, intermediate))
    route_values = tuple(index % experts for index in range(rows * selected))
    score_values = tuple(1.0 / selected for _ in route_values)
    return KernelCase(
        lambda value, indices, weights, gate_weight, up_weight, down_weight: mt.routed_experts(
            value, indices, weights, gate_weight, up_weight, down_weight
        ),
        (
            Operand(mt.Argument(hidden, "hidden")),
            Operand(
                mt.Argument(routes, "routes"),
                struct.pack(f"={len(route_values)}i", *route_values),
            ),
            Operand(
                mt.Argument(scores, "scores"),
                struct.pack(f"={len(score_values)}f", *score_values),
            ),
            Operand(mt.Argument(gate, "gate", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(gate, "up", mt.ValueKind.CONSTANT)),
            Operand(mt.Argument(down, "down", mt.ValueKind.CONSTANT)),
        ),
    )


def definition(
    name: str,
    rows: int,
    context: int,
    *,
    width: int = 1024,
    output: int = 3072,
    intermediate: int = 3072,
) -> KernelCase:
    return {
        "encoded-linear": lambda: _linear(rows, width, output),
        "parallel-linear": lambda: _parallel_linear(rows, width, output),
        "dense-swiglu": lambda: _dense_swiglu(rows, width, intermediate),
        "attention": lambda: _attention(rows, context),
        "recurrent-prepare": lambda: _recurrent_prepare(rows),
        "gated-recurrence": lambda: _gated_recurrence(rows),
        "grouped-experts": lambda: _experts(rows),
    }[name]()


def run(
    name: str,
    *,
    backend: str,
    mode: str,
    rows: int,
    context: int,
    warmup: int,
    repeat: int,
    memory_bytes: int,
    width: int = 1024,
    output: int = 3072,
    intermediate: int = 3072,
) -> KernelResult:
    case = definition(
        name,
        rows,
        context,
        width=width,
        output=output,
        intermediate=intermediate,
    )
    device = mt.device(backend, budget_bytes=memory_bytes)
    resources = []
    constants = {}
    mutable = {}
    dynamic = []
    try:
        for operand in case.operands:
            spec = operand.argument.spec
            resource = (
                device.allocate(spec)
                if operand.content is None
                else device.upload(spec, operand.content)
            )
            resources.append(resource)
            if operand.argument.kind == mt.ValueKind.CONSTANT:
                constants[operand.argument.name] = resource
            elif operand.argument.kind == mt.ValueKind.RESOURCE:
                mutable[operand.argument.name] = resource
            else:
                dynamic.append(resource)
        compiled = mt.compile(
            case.function,
            signature=mt.Signature(tuple(item.argument for item in case.operands)),
            device=device,
            constants=constants,
            options=mt.CompileOptions(mode=mode),
        )
        try:
            samples = []
            for iteration in range(warmup + repeat):
                started = time.perf_counter_ns()
                execution = compiled.submit(*dynamic, resources=mutable)
                execution.completion.wait()
                elapsed = (time.perf_counter_ns() - started) / 1e6
                for result_resource in execution.outputs:
                    result_resource.close()
                if iteration >= warmup:
                    samples.append(elapsed)
            selected = tuple(name for unit in compiled.diagnostics.submissions for name in unit)
            return KernelResult(
                name,
                selected,
                compiled.diagnostics.dispatches,
                warmup,
                repeat,
                statistics.median(samples),
                min(samples),
                max(samples),
            )
        finally:
            compiled.close()
    finally:
        for resource in reversed(resources):
            resource.close()
        device.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=(
            "encoded-linear",
            "parallel-linear",
            "dense-swiglu",
            "attention",
            "recurrent-prepare",
            "gated-recurrence",
            "grouped-experts",
        ),
    )
    parser.add_argument("--backend", default="metal", choices=("metal", "cuda", "hip", "llvm"))
    parser.add_argument("--mode", default="prefill", choices=("prefill", "decode"))
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output", type=int, default=3072)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-bytes", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    if (
        min(
            args.rows,
            args.context,
            args.width,
            args.output,
            args.intermediate,
            args.repeat,
            args.memory_bytes,
        )
        <= 0
        or args.warmup < 0
    ):
        parser.error("row, context, repeat and memory values must be positive")
    result = run(
        args.case,
        backend=args.backend,
        mode=args.mode,
        rows=args.rows,
        context=args.context,
        warmup=args.warmup,
        repeat=args.repeat,
        memory_bytes=args.memory_bytes,
        width=args.width,
        output=args.output,
        intermediate=args.intermediate,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
