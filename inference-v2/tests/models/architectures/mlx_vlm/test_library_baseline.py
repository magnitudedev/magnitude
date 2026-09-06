from dataclasses import replace

import mlx.core as mx
import pytest

from magnitude_engine import blueprints as bp
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.worker.host import Worker
from tests.models.architectures.qwen35.test_construction import artifact_pair
from tests.worker.test_worker import collect, configuration


def test_library_artifact_runs_native_cache_and_preserves_branch_isolation(tmp_path):
    path, _, oracle, _ = artifact_pair(tmp_path)
    engine_bp = bp.engine.Engine(
        generation=bp.generation.Generation(
            target=bp.model.auto(bp.model.artifacts.Local(path=str(path)))
        ),
        memory=bp.engine.memory.Budgeted(limit_bytes=64 << 20),
    )
    lifetime = bp.build(engine_bp)
    engine = lifetime.__enter__()
    budget = engine.budget
    runtime = engine.engine.generation.model
    loaded = runtime.program
    row = runtime.create()
    runtime.prefill(row, (1, 2, 3))
    checkpoint = row.checkpoint()
    sibling = runtime.create(checkpoint)
    with pytest.raises(RuntimeError, match="active sequences"):
        loaded.close()
    for sequence, tokens, accepted in ((row, (4, 5, 6), 1), (sibling, (9, 10), 2)):
        advance = runtime.forward(sequence, tokens)
        advance.complete()
        expected = oracle(mx.array([[1, 2, 3, *tokens]]))[:, -len(tokens) :]
        assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
        advance.accept(accepted)
        follow = runtime.forward(sequence, (11,))
        follow.complete()
        expected = oracle(mx.array([[1, 2, 3, *tokens[:accepted], 11]]))[:, -1:]
        assert mx.allclose(follow.output.logits, expected, atol=1e-4, rtol=1e-4).item()
        follow.accept(1)
        sequence.close()
    checkpoint.close()
    lifetime.__exit__(None, None, None)
    assert budget.snapshot().reserved == 0


def test_library_worker_reports_resident_baseline_and_reuses_prefix(tmp_path):
    previous = configuration(tmp_path)
    artifact = previous.generation.target.program.artifact
    config = replace(previous, generation=bp.generation.Generation(target=bp.model.auto(artifact)))
    with Worker(config, startup_timeout=20) as host:
        assert host.properties["program_implementation"].startswith("mlx_vlm.")
        assert host.properties["composition_digest"] == bp.digest(config)
        assert host.properties["speculative_backend"] is None
        first, cold = collect(host.submit((1, 2, 3), SamplingPolicy(temperature=0), 6))
        repeated, warm = collect(host.submit((1, 2, 3), SamplingPolicy(temperature=0), 6))
        assert first == repeated and len(first) == 6
        assert cold.cached_tokens == 0
        assert warm.cached_tokens == 2
