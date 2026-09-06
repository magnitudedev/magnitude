"""Opt-in real hybrid artifact check; keep costly model loading out of routine tests."""

import os
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.qwen3_5 import gated_delta_update
from mlx_lm.utils import load_model

from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.models.architectures.qwen35.attention.binding import Attention
from magnitude_engine.models.architectures.qwen35.binding import bind_qwen35
from magnitude_engine.models.architectures.qwen35.feedforward.binding import MoE
from magnitude_engine.models.architectures.qwen35.recurrence.binding import Mixer
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.experts.binding import Resident as ResidentExpertFactory
from magnitude_engine.models.recurrence.metal import MetalDelta
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.arena import KVArena
from magnitude_engine.models.state.hybrid import HybridStateStore
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.resources.budget import MemoryBudget


@pytest.mark.model
def test_local_dense_hybrid_rejection_and_continuation_match_library():
    directory = os.environ.get("MAGNITUDE_TEST_QWEN35")
    if not directory:
        pytest.skip("set MAGNITUDE_TEST_QWEN35 to a local dense Qwen3.5 MLX artifact")
    loaded, _ = load_model(Path(directory))
    model = loaded.language_model
    assert not model.args.num_experts
    embed = model.model.embed_tokens
    binding = bind_qwen35(
        model,
        embedding=ResidentAffineEmbedding(
            embed.weight, embed.scales, embed.biases, AffineEncoding(embed.bits, embed.group_size)
        ),
        experts={},
        attention=Attention(GatheredAttention()),
        recurrence=Mixer(MetalDelta()),
        feedforward=MoE(ResidentExpertFactory()),
        state_dtype=embed.scales.dtype,
    )
    budget = MemoryBudget(2 << 30)
    arena = KVArena(
        binding.attention,
        page_size=16,
        slab_pages=16,
        max_pages=64,
        budget=budget,
        dtype=embed.scales.dtype,
    )
    runtime = ModelRuntime(
        binding.program,
        HybridStateStore(PageStore(arena), binding.recurrence, budget),
        ExecutionOwner(),
    )
    row = runtime.create()
    runtime.prefill(row, (10, 20, 30))
    verify = runtime.forward(row, (40, 50, 60))
    matching_cache = model.make_cache()
    mx.eval(model(mx.array([[10, 20, 30]]), cache=matching_cache))
    recurrent_indices = [i for i, layer in enumerate(model.layers) if layer.is_linear]
    windows = {i: mx.array(matching_cache[i][0]) for i in recurrent_indices}
    prepared = []

    def record_recurrence(*args, **kwargs):
        prepared.append((args, kwargs))
        return gated_delta_update(*args, **kwargs)

    # The oracle records inputs to the library recurrence. Replaying these inputs
    # preserves verification-shape projection rounding, as the PoC does. Re-running
    # a shorter transformer block is a different numerical experiment in BF16.
    with patch("mlx_lm.models.qwen3_5.gated_delta_update", record_recurrence):
        matching = model(mx.array([[40, 50, 60]]), cache=matching_cache)
    assert mx.array_equal(verify.output.logits, matching).item()
    for index, (args, kwargs) in zip(recurrent_indices, prepared, strict=True):
        prefix_args = (*[a[:, :1] for a in args[:5]], *args[5:])
        _, memory = gated_delta_update(*prefix_args, **kwargs)
        previous = windows[index]
        assert previous.shape[1] >= 3
        raw = matching_cache[index][0][:, -3:]
        window = mx.array(mx.concatenate([previous, raw], axis=1)[:, 1 : 1 + previous.shape[1]])
        matching_cache[index].state = [window, memory]
    for index, cache in enumerate(matching_cache):
        if index not in recurrent_indices:
            assert cache.trim(2) == 2
    verify.accept(1)
    continuation = runtime.forward(row, (70,))
    continuation.complete()
    actual = continuation.output.logits[0, -1].astype(mx.float32)
    expected = model(mx.array([[70]]), cache=matching_cache)[0, -1].astype(mx.float32)
    mx.eval(actual, expected)
    maximum = float(mx.max(mx.abs(actual - expected)).item())
    relative = float((mx.linalg.norm(actual - expected) / mx.linalg.norm(expected)).item())
    assert mx.argmax(actual).item() == mx.argmax(expected).item()
    assert maximum < 0.5 and relative < 0.01
    print(
        {
            "artifact": directory,
            "max_logit_error": maximum,
            "relative_l2_error": relative,
            "selected_token": actual.argmax().item(),
            "verify_bit_identical": True,
        }
    )
    row.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0
