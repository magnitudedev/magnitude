"""Benchmark a component constructed from the public blueprint graph."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from magnitude_engine.composition import build, digest, loads
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.operations.attention import CausalAttention
from magnitude_engine.operations.linear import ResidentLinear
from magnitude_engine.operations.recurrent import DeltaRecurrence
from magnitude_engine.platform.host.machine import discover
from magnitude_engine.platform.host.measurement import exclusive_measurement
from performance import benchmarks
from performance.attention import AttentionWorkload
from performance.linear import LinearWorkload
from performance.model import ModelWorkload
from performance.recurrent import RecurrentWorkload
from performance.runner import Policy
from performance.thermals import ThermalRecorder


def source_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            *root.joinpath("src/magnitude_engine").rglob("*.py"),
            *root.joinpath("performance").rglob("*.py"),
            root / "pyproject.toml",
            root / "uv.lock",
        )
    ):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blueprint", type=Path)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--host-profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = loads(args.blueprint.read_text())
    root = Path(__file__).resolve().parent.parent
    source = source_identity(root)
    workload_json = args.workload.read_text()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = args.output.parent / (args.output.stem + ".evidence")
    evidence.mkdir(parents=True)
    with exclusive_measurement(), ThermalRecorder(evidence) as thermal, build(recipe) as component:
        machine = discover()
        policy = Policy(repetitions=args.repetitions, host_profile=args.host_profile)
        if isinstance(component, ResidentLinear):
            workload = LinearWorkload.model_validate_json(workload_json)
            result = benchmarks.run(component, workload, policy=policy)
            artifact_identity = None
        elif isinstance(component, DenseRuntime):
            workload = ModelWorkload.model_validate_json(workload_json)
            result = benchmarks.run(component, workload, policy=policy)
            artifact_identity = None
        elif isinstance(component, DeltaRecurrence):
            workload = RecurrentWorkload.model_validate_json(workload_json)
            result = benchmarks.run(component, workload, policy=policy)
            artifact_identity = None
        elif isinstance(component, CausalAttention):
            workload = AttentionWorkload.model_validate_json(workload_json)
            result = benchmarks.run(component, workload, policy=policy)
            artifact_identity = None
        else:
            raise TypeError("no benchmark procedure has yet been admitted for this component type")
        from magnitude_engine.platform.compiler import compiler_build

        record = {
            "compiler": compiler_build().model_dump(mode="json"),
            "blueprint": json.loads(args.blueprint.read_text()),
            "blueprint_digest": digest(recipe),
            "source_digest": source,
            "machine": machine.model_dump(mode="json"),
            "runtime": {
                name: importlib.metadata.version(name)
                for name in ("tilelang", "torch", "apache-tvm-ffi", "gguf")
            },
            "artifact_sha256": artifact_identity,
            "workload": workload.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        }
    record["thermals"] = thermal.summary
    record["thermal_evidence_directory"] = evidence.name
    if source_identity(root) != source:
        raise RuntimeError("source changed during the benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    latency = result.metrics.completed_latency.median_seconds
    device = result.metrics.device_latency.median_seconds
    print(f"validation: {'pass' if result.validation.passed else 'FAIL'}")
    print(
        f"completed median: {latency * 1e6:.1f} us" if latency is not None else "no latency samples"
    )
    print(
        f"device median: {device * 1e6:.1f} us"
        if device is not None
        else "device timing unavailable"
    )
    print(f"saved: {args.output}")
    if result.host_profile is not None:
        print("Host call profile (separate instrumented invocation; own time):")
        for call in sorted(
            result.host_profile.calls, key=lambda call: call.own_seconds, reverse=True
        )[:10]:
            print(
                f"  {call.own_seconds * 1e3:.3f} ms, {call.total_calls} calls: "
                f"{call.site.file}:{call.site.line} {call.site.function}"
            )
    if not result.validation.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
