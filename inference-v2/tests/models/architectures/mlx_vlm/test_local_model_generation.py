"""Opt-in local artifact integration; no network or model acquisition in tests."""

import os
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model, load_tokenizer

from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget


@pytest.mark.model
def test_local_qwen3_greedy_generation_and_checkpoint_match_library():
    path = os.environ.get("MAGNITUDE_TEST_QWEN3")
    if not path:
        pytest.skip("set MAGNITUDE_TEST_QWEN3 to a local Qwen3 artifact directory")
    artifact = Path(path)
    model, config = load_model(artifact)
    assert config["model_type"] == "qwen3"
    tokenizer = load_tokenizer(artifact)
    prompt = tuple(tokenizer.encode("The capital of France is"))
    budget = MemoryBudget(256 << 20)
    # This fixture's resident KV geometry is known; production construction supplies
    # family-specific estimators for other state and quantization layouts.
    token_bytes = (
        config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * 2 * 4
    )
    states = LibraryStateStore(
        lambda: make_prompt_cache(model), budget, lambda n: ((n + 255) // 256) * 256 * token_bytes
    )
    runtime = ModelRuntime(
        LibraryProgram(lambda ids, cache: model(ids, cache=cache)), states, ExecutionOwner()
    )
    generation = GenerationRuntime(runtime, SuffixMethod())
    row = generation.create(prompt, SamplingPolicy(temperature=0), 12, chunk_size=3)
    generated = []
    checkpoint = row.checkpoint()
    while not row.finished:
        generated.extend(row.step(5).tokens)
    cache = make_prompt_cache(model)
    inputs = mx.array([prompt])
    expected = []
    for _ in range(12):
        token = int(mx.argmax(model(inputs, cache=cache)[0, -1]).item())
        expected.append(token)
        inputs = mx.array([[token]])
    assert generated == expected
    restored = generation.create(prompt, SamplingPolicy(temperature=0), 12, checkpoint=checkpoint)
    actual = []
    while not restored.finished:
        actual.extend(restored.step(5).tokens)
    assert actual == expected
    print({"artifact": str(artifact), "tokens": generated, "text": tokenizer.decode(generated)})
    restored.close()
    checkpoint.close()
    row.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0
