"""Bounded timing attribution of actual model-selected regions, using synthetic data.

This is a diagnostic, not numerical qualification or a model throughput result.
It preserves production shapes, packet formats and selected emitters. Uniform
routes and zero-valued operands deliberately isolate schedule cost from model
loading; they do not reproduce real expert occupancy or cache competition.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import time
from collections import Counter
from pathlib import Path

import magnitensor as mt
from magnitensor.compiler.lowering import SubmissionUnit
from magnitensor.compiler.unit import build_unit
from magnitude_engine.models.qwen35.tensor_program import define, weight_roles
from magnitude_engine.qualification import QualificationCase, invocation_specs
from magnitude_engine.weights.formats.gguf import GGUFFormat
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat
from magnitude_engine.weights.tensor_residency import TensorWeights

FAMILIES = (
    "attention.matrix-streaming-gated-output",
    "attention.register-partitioned-gated-output",
    "gated_delta.register-state",
    "gated_delta.chunked-matrix",
    "linear.parallel-packet-prefill",
    "linear.parallel-packet-decode",
    "dense_swiglu.packet-prefill",
    "dense_swiglu.packet-decode",
    "routed_experts.grouped",
    "routed_experts.packet-shared",
)


def measure(device, plan, candidate, rows, context):
    unit = build_unit(plan.graph, plan.memory, SubmissionUnit(0, (candidate,), "attribution"))
    call = unit.calls[0]
    content = {}
    family = candidate.name.split("@", 1)[0]
    if family.startswith("attention."):
        limits = tuple(value for _ in range(rows) for value in (0, context))
        content[call.bindings[2].parameter] = struct.pack(f"={len(limits)}i", *limits)
    elif family.startswith("gated_delta."):
        content[call.bindings[6].parameter] = struct.pack("=ii", 0, rows)
    elif family.startswith("routed_experts."):
        specs = candidate.emitter.specs
        selected = specs[1].shape[1]
        experts = specs[3].shape[0]
        indices = tuple(index % experts for index in range(rows * selected))
        content[call.bindings[1].parameter] = struct.pack(f"={len(indices)}i", *indices)
        content[call.bindings[2].parameter] = struct.pack(
            f"={len(indices)}f", *(1.0 / selected for _ in indices)
        )
    resources = []
    executable = entrypoint = None
    try:
        for parameter in unit.parameters:
            resources.append(
                device.upload(
                    parameter.spec,
                    content.get(parameter.name, bytes(parameter.spec.storage_nbytes)),
                )
            )
        executable = device.runtime.compile(unit, unit.signature)
        entrypoint = executable.bind(
            {index: resource.native for index, resource in enumerate(resources)}, ()
        )
        samples = []
        for iteration in range(4):
            started = time.perf_counter_ns()
            entrypoint.submit(()).wait()
            if iteration:
                samples.append((time.perf_counter_ns() - started) / 1e6)
        return statistics.median(samples)
    finally:
        if entrypoint is not None:
            entrypoint.close()
        if executable is not None:
            executable.close()
        for resource in reversed(resources):
            resource.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefill", "decode"))
    parser.add_argument("--family", choices=FAMILIES, action="append")
    args = parser.parse_args()
    records = []
    for path in args.target:
        format = MLXFormat(str(path)) if path.is_dir() else GGUFFormat(str(path))
        device = mt.device("auto", budget_bytes=32 * 1024**3)
        residency = TensorWeights(format, device)
        try:
            if isinstance(format, MLXFormat):
                from magnitude_engine.models.qwen35.formats.mlx import describe as describe_mlx

                description = describe_mlx(format)
            else:
                from magnitude_engine.models.qwen35.formats.gguf import describe as describe_gguf

                description = describe_gguf(format)
            weight_specs = {
                descriptor.name: residency.spec(descriptor, dtype)
                for descriptor, dtype in weight_roles(description)
            }
            for mode, rows in (("prefill", 2048), ("decode", 1)):
                if args.mode is not None and args.mode != mode:
                    continue
                case = QualificationCase(mode, mode, rows, 1, 65792, True)
                definition = define(
                    description, weight_specs, mode, invocation_specs(description, case, slots=1)
                )
                plan = mt.analyze(
                    definition.function,
                    signature=definition.signature,
                    capabilities=device.capabilities,
                    compiler_identity=device.compiler_identity,
                    available_bytes=device.available_bytes,
                    options=definition.options,
                )
                selected = {}
                counts = Counter()
                for candidate in plan.cover.candidates:
                    family = candidate.name.split("@", 1)[0]
                    if family not in (args.family or FAMILIES):
                        continue
                    key = (family, repr(vars(candidate.emitter)))
                    selected.setdefault(key, candidate)
                    counts[key] += 1
                for key, candidate in selected.items():
                    elapsed = measure(device, plan, candidate, rows, 65536)
                    record = {
                        "target": str(path),
                        "mode": mode,
                        "rows": rows,
                        "context": 65536,
                        "family": key[0],
                        "occurrences": counts[key],
                        "median_ms": elapsed,
                        "sum_region_ms": elapsed * counts[key],
                        "kernels": candidate.kernel_count,
                        "input_specs": [repr(plan.graph.values[i].spec) for i in candidate.inputs],
                    }
                    records.append(record)
                    args.output.write_text(json.dumps(records, indent=2) + "\n")
                    print(
                        json.dumps({k: v for k, v in record.items() if k != "input_specs"}),
                        flush=True,
                    )
        finally:
            residency.close()
            device.close()
            format.close()


if __name__ == "__main__":
    main()
