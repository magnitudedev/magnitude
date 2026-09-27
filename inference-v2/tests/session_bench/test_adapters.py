import hashlib
from pathlib import Path

import pytest

from session_bench.engines import ADAPTERS
from session_bench.engines.omlx_native import instrumentation
from session_bench.models import Target, prepare
from session_bench.results import RunStore


@pytest.mark.parametrize("engine", ["magnitude", "mlx-vlm", "omlx", "llama.cpp"])
def test_readiness_requires_matching_loaded_capacity(engine, artifact_path, tmp_path):
    # Readiness checks stay lightweight and don't load an engine or a model.
    target = Target(engine=engine, reference=str(artifact_path))
    artifact = prepare(Target(engine="magnitude", reference=str(artifact_path)))
    root = Path(__file__).resolve().parents[2]
    adapter = ADAPTERS[engine](root, target, artifact, RunStore(tmp_path, "test", {}))
    if engine == "magnitude":
        data = {
            "status": "ready",
            "model": "session-bench",
            "context_tokens": 40000,
            "parallel_sequences": 4,
            "speculative_backend": "none",
            "retained_prefixes": 0,
            "prefill_tokens": 512,
            "output_capacity": 64,
            "memory_bytes": adapter.memory_limit,
            "composition_json": "{}",
            "composition_digest": hashlib.sha256(b"{}").hexdigest(),
        }
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
