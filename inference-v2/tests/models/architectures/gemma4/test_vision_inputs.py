"""Image operands against upstream layers with independently constructed HF masks."""

import mlx.core as mx
import pytest

from magnitude_engine.models.architectures.gemma4.inputs import GemmaInputs
from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.embeddings.replacement import EmbeddingReplacement
from magnitude_engine.models.inputs import ModelInputs
from tests.models.architectures.gemma4.test_gemma_model import compose_gemma


def reference(model, tokens, start, features, *, bidirectional=True):
    inner = model.model
    count = tokens.shape[1]
    end = start + features.shape[1]
    hidden = inner.embed_tokens(tokens) * inner.embed_scale
    hidden[:, start:end] = features
    per_layer = None
    if inner.hidden_size_per_layer_input:
        lookup_ids = mx.array(tokens)
        lookup_ids[:, start:end] = 0
        per_layer = inner._project_per_layer_inputs(hidden, inner._get_per_layer_inputs(lookup_ids))
    queries, keys = mx.arange(count)[:, None], mx.arange(count)[None]
    causal = keys <= queries
    image = (queries >= start) & (queries < end) & (keys >= start) & (keys < end)
    local = (causal | image if bidirectional else causal) & (keys > queries - inner.window_size)
    saved = [(None, None)] * len(inner.layers)
    for index, (layer, previous) in enumerate(zip(inner.layers, inner.previous_kvs, strict=True)):
        shared, offset = saved[previous]
        hidden, shared, offset = layer(
            hidden,
            local if layer.layer_type == "sliding_attention" else causal,
            None,
            per_layer_input=None if per_layer is None else per_layer[:, :, index],
            shared_kv=shared,
            offset=offset,
        )
        saved[index] = shared, offset
    logits = inner.embed_tokens.as_linear(inner.norm(hidden))
    return mx.tanh(logits / 13.0) * 13.0


@pytest.mark.parametrize("visibility", [(True, True), (False, False), (False, True)])
@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("shared", [True, False])
def test_image_prefill_and_compiled_continuation_preserve_all_gemma_branches(
    bits, shared, visibility, monkeypatch
):
    attention = MetalPagedAttention()
    gathered = attention.prefill.compute
    fallbacks = []

    def record_fallback(*args, **kwargs):
        fallbacks.append(kwargs.get("key_ends"))
        return gathered(*args, **kwargs)

    monkeypatch.setattr(attention.prefill, "compute", record_fallback)
    model, runtime, arena, budget = compose_gemma(bits=bits, shared=shared, attention=attention)
    prefixes = ((1, 2, 3), tuple(range(1, 20)))
    rows = tuple(runtime.create() for _ in prefixes)
    features = (mx.random.normal((1, 4, 64)), mx.random.normal((1, 4, 64)))
    tokens = (20, 21, 21, 21, 21, 22)
    for row, prefix in zip(rows, prefixes, strict=True):
        runtime.prefill(row, prefix)
    inputs = tuple(
        ModelInputs(
            mx.array([tokens], mx.int32),
            data=GemmaInputs(
                (EmbeddingReplacement(1, image),),
                mx.array([[True, False, False, False, False, True]]),
                mx.array([[len(prefix) + 1, *([len(prefix) + 5] * 4), len(prefix) + 6]], mx.int32)
                if bidirectional
                else None,
            ),
        )
        for prefix, image, bidirectional in zip(prefixes, features, visibility, strict=True)
    )
    fallbacks.clear()
    advances = runtime.forward_batch(rows, inputs)
    assert bool(fallbacks) == any(visibility)
    for advance, prefix, image, bidirectional in zip(
        advances, prefixes, features, visibility, strict=True
    ):
        advance.complete()
        expected = reference(
            model,
            mx.array([[*prefix, *tokens]], mx.int32),
            len(prefix) + 1,
            image,
            bidirectional=bidirectional,
        )
        assert mx.allclose(advance.output.logits, expected[:, -6:], atol=1e-4, rtol=1e-4).item()
        advance.accept(6)
    checkpoints = tuple(row.checkpoint() for row in rows)
    restored = tuple(runtime.create(checkpoint) for checkpoint in checkpoints)
    for row, prefix, image, bidirectional in zip(
        restored, prefixes, features, visibility, strict=True
    ):
        advance = runtime.forward(row, (23,))
        advance.complete()
        expected = reference(
            model,
            mx.array([[*prefix, *tokens, 23]], mx.int32),
            len(prefix) + 1,
            image,
            bidirectional=bidirectional,
        )
        assert mx.allclose(advance.output.logits, expected[:, -1:], atol=1e-4, rtol=1e-4).item()
        advance.accept(1)
    for row in (*rows, *restored):
        row.close()
    for checkpoint in checkpoints:
        checkpoint.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0
