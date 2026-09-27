from pathlib import Path

import pytest

from magnitude_benchmarks.adapters import ADAPTERS
from magnitude_benchmarks.adapters.native import engine_sources, sources_digest
from magnitude_benchmarks.adapters.omlx_native import instrumentation
from magnitude_benchmarks.session_bench.models import Target, prepare
from magnitude_benchmarks.session_bench.options import EngineOptions, NativeOptions
from magnitude_benchmarks.session_bench.policy import project_root
from magnitude_benchmarks.session_bench.results import RunStore


@pytest.mark.parametrize("engine", ["magnitude", "mlx-vlm", "omlx", "llama.cpp"])
def test_readiness_requires_matching_loaded_capacity(engine, artifact_path, tmp_path):
    # Readiness checks stay lightweight and don't load an engine or a model.
    target = Target(engine=engine, reference=str(artifact_path))
    artifact = prepare(Target(engine="mlx-vlm", reference=str(artifact_path)))
    adapter = ADAPTERS[engine](
        project_root(), target, artifact, RunStore(tmp_path, "test", {}), EngineOptions()
    )
    if engine == "magnitude":
        data = {"ready": True, "model": "session-bench", "context_tokens": 40000, "vocabulary": 9}
    elif engine == "mlx-vlm":
        data = {
            "status": "healthy",
            "loaded_model": str(artifact_path),
            "effective_context_limit": 40000,
            "apc_enabled": False,
            "continuous_batching_enabled": True,
        }
    elif engine == "omlx":
        data = {
            "ready": True,
            "served_model": "session-bench",
            "loaded": True,
            "context_capacity": 40000,
            "max_concurrent_requests": 4,
            "speculative_backend": "none",
        }
    else:
        data = {"total_slots": 4, "default_generation_settings": {"n_ctx": 40000}}
    adapter.verify_ready(data, 40000, 4)
    with pytest.raises(ValueError):
        adapter.verify_ready(data, 41000, 4)


def test_native_launch_passes_engine_selection(artifact_path, tmp_path):
    target = Target(engine="magnitude", reference=str(artifact_path))
    artifact = prepare(Target(engine="mlx-vlm", reference=str(artifact_path)))
    options = EngineOptions(
        native=NativeOptions(
            binary=tmp_path / "engine", device="cpu", method="mtp", mtp_proposals=2
        )
    )
    adapter = ADAPTERS["magnitude"](
        project_root(), target, artifact, RunStore(tmp_path, "test", {}), options
    )
    argv = adapter.argv(1234, 8192, 2, tmp_path)
    assert argv[0] == str(tmp_path / "engine")
    def flag(name):
        return argv[argv.index(name) + 1]

    assert "--no-projector" in argv
    assert flag("--context-tokens") == "8192"
    assert "--max-batch" not in argv
    assert flag("--device") == "cpu" and flag("--mtp-proposals") == "2"
    with pytest.raises(ValueError, match="requires --native-method mtp"):
        NativeOptions(mtp_proposals=2)


def write(root: Path, name: str, content: str = "before") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_native_source_identity_covers_build_inputs_only(tmp_path):
    cpp = write(tmp_path, "engine/templates/native/src/abi.cpp")
    header = write(tmp_path, "engine/templates/native/include/templates.h")
    write(tmp_path, "engine/templates/build.rs")
    write(tmp_path, "engine/src/lib.rs")
    write(tmp_path, "Cargo.lock")
    write(tmp_path, "target/release/output.rs", "ignored")
    write(tmp_path, "benchmarks/src/runner.rs", "ignored")
    evidence = engine_sources(tmp_path)
    assert set(evidence) == {
        "Cargo.lock",
        "engine/src/lib.rs",
        "engine/templates/build.rs",
        "engine/templates/native/include/templates.h",
        "engine/templates/native/src/abi.cpp",
    }
    baseline = sources_digest(evidence)
    cpp.write_text("after")
    assert sources_digest(engine_sources(tmp_path)) != baseline
    cpp.write_text("before")
    header.write_text("after")
    assert sources_digest(engine_sources(tmp_path)) != baseline


def test_omlx_native_enrichment_keeps_evidence_and_checks_phase_boundary():
    metric = instrumentation.RequestMetrics(prompt_ms=10)
    instrumentation._record_batched_service_time(metric, 5, has_uncached_prompt_token=True)
    instrumentation._record_batched_service_time(metric, 7, has_uncached_prompt_token=False)
    assert metric.prompt_ms == 15
    assert metric.generation_ms == 7
    key = "test-native-timing"
    instrumentation._metrics[key] = metric
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    try:
        value = instrumentation._terminal_payload({"usage": usage, "choices": []}, key)
        assert value["usage"] == usage
        assert value["timings"]["predicted_ms"] == 7
        usage["total_tokens"] = 99
        with pytest.raises(RuntimeError, match="disagrees"):
            instrumentation._terminal_payload({"usage": usage}, key)
    finally:
        instrumentation._metrics.pop(key)
