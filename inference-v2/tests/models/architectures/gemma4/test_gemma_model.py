import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.gemma4_text.config import ModelConfig
from mlx_vlm.models.gemma4_text.language import LanguageModel

from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.kernels.contractions.weights import ExpertWeights, QuantizedProjection
from magnitude_engine.models.architectures.gemma4.binding import bind_gemma4
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding, ResidentEmbedding
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.experts.computation import GatedExpertMath, ResidentExperts
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelRuntime
from magnitude_engine.models.state.arena import KVArena
from magnitude_engine.models.state.paged import PagedStateStore
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.resources.budget import MemoryBudget


def embedding_operation(module):
    if hasattr(module, "scales"):
        return ResidentAffineEmbedding(
            module.weight,
            module.scales,
            module.biases,
            AffineEncoding(module.bits, module.group_size),
        )
    return ResidentEmbedding(module.weight)


def compose_gemma(*, bits, shared, attention, global_width=64):
    mx.random.seed(352)
    config = ModelConfig(
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=32,
        global_head_dim=global_width,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=32 if shared else 0,
        num_kv_shared_layers=2 if shared else 0,
        sliding_window=7,
        sliding_window_pattern=2,
        attention_k_eq_v=True,
        enable_moe_block=bits is not None,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=96,
        final_logit_softcapping=13.0,
    )
    model = LanguageModel(config)
    for index, layer in enumerate(model.layers):
        layer.layer_scalar = mx.array([0.8 + index * 0.07])
        if bits is not None:
            layer.router.scale = mx.linspace(0.7, 1.2, 64)
            layer.router.per_expert_scale = mx.linspace(0.4, 1.7, 8)
    if bits is not None:
        nn.quantize(model, group_size=32, bits=bits)
    # Match loaded artifacts: parameters are fixed values, not a lazy random
    # quantization graph that compilation can fuse with neural execution.
    mx.eval(model.parameters())
    experts = {}
    for index, layer in enumerate(model.layers):
        if not layer.enable_moe:
            continue

        def projection(m):
            return QuantizedProjection(
                m.weight, m.scales, m.biases, AffineEncoding(m.bits, m.group_size)
            )

        switch = layer.experts.switch_glu
        experts[index] = ResidentExperts(
            ExpertWeights(
                projection(switch.up_proj),
                projection(switch.gate_proj),
                projection(switch.down_proj),
            ),
            GatedExpertMath(lambda up, gate: nn.gelu_approx(gate) * up),
        )
    binding = bind_gemma4(
        model,
        embedding=embedding_operation(model.model.embed_tokens),
        per_layer_embedding=(
            embedding_operation(model.model.embed_tokens_per_layer) if shared else None
        ),
        experts=experts,
        attention=attention,
    )
    budget = MemoryBudget(32 << 20)
    arena = KVArena(
        binding.attention,
        page_size=4,
        slab_pages=4,
        max_pages=128,
        budget=budget,
        dtype=mx.float32,
    )
    runtime = ModelRuntime(binding.program, PagedStateStore(PageStore(arena)), ExecutionOwner())
    return model, runtime, arena, budget


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("accepted", [0, 2, 4])
def test_gemma_shared_state_windowed_batch_reconciliation(bits, shared, accepted):
    model, runtime, arena, budget = compose_gemma(
        bits=bits, shared=shared, attention=MetalPagedAttention()
    )
    assert len(arena.layers) == (2 if shared else 4)
    prefixes = [tuple(range(1, 4)), tuple(range(1, 20))]
    rows = tuple(runtime.create() for _ in prefixes)
    for row, prefix in zip(rows, prefixes, strict=True):
        runtime.prefill(row, prefix)
    checkpoint = rows[1].checkpoint()
    branch = runtime.create(checkpoint)
    tokens = (21, 22, 23, 24)
    advances = runtime.forward_batch(
        rows,
        tuple(ModelInputs.from_tokens(tokens) for _ in rows),
        ForwardRequest(features=frozenset({"residual:4"})),
    )
    for advance, prefix in zip(advances, prefixes, strict=True):
        advance.complete()
        expected = model(mx.array([[*prefix, *tokens]])).logits[:, -len(tokens) :]
        assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
        assert advance.output.features["residual:4"].shape == (1, 4, 64)
        # Slab reconciliation must never rerun a neural block.
        original = runtime.program.forward
        runtime.program.forward = lambda *a, **k: pytest.fail("unexpected neural replay")
        advance.accept(accepted)
        runtime.program.forward = original
    for row, prefix in zip(rows, prefixes, strict=True):
        step = runtime.forward(row, (31, 32))
        step.complete()
        expected = model(mx.array([[*prefix, *tokens[:accepted], 31, 32]])).logits[:, -2:]
        assert mx.allclose(step.output.logits, expected, atol=1e-4, rtol=1e-4).item()
        step.accept(2)
        row.close()
    step = runtime.forward(branch, (41,))
    step.complete()
    expected = model(mx.array([[*prefixes[1], 41]])).logits[:, -1:]
    assert mx.allclose(step.output.logits, expected, atol=1e-4, rtol=1e-4).item()
    step.accept(1)
    branch.close()
    checkpoint.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("attention", [GatheredAttention(), MetalPagedAttention()])
def test_gemma_full_width_attention_and_prefill_without_vocabulary_projection(attention):
    model, runtime, arena, budget = compose_gemma(
        bits=4, shared=False, attention=attention, global_width=512
    )
    row = runtime.create()
    original = runtime.program.output
    runtime.program.output = lambda *a: pytest.fail("prefill projected the vocabulary")
    runtime.prefill(row, (1, 2, 3, 4, 5))
    runtime.program.output = original
    advance = runtime.forward(row, (6, 7))
    advance.complete()
    expected = model(mx.array([[1, 2, 3, 4, 5, 6, 7]])).logits[:, -2:]
    assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
    advance.accept(2)
    row.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_gemma_swaps_both_embedding_tables_and_experts_through_owned_contracts(tmp_path):
    from magnitude_engine.artifacts.layouts import logical_tensors
    from magnitude_engine.artifacts.tensors import TensorCatalog
    from magnitude_engine.models.embeddings.streaming import StreamedEmbedding
    from magnitude_engine.models.embeddings.table import AffineRowTable
    from magnitude_engine.models.experts.bank import ExpertBank, ExpertSource, ProjectionSource
    from magnitude_engine.models.experts.streaming import StreamedExperts
    from magnitude_engine.resources.io.reader import PositionalReader

    model, resident, arena, budget = compose_gemma(
        bits=4, shared=True, attention=MetalPagedAttention()
    )
    components = ("weight", "scales", "biases")
    tables = {
        "tokens": model.model.embed_tokens,
        "per_layer": model.model.embed_tokens_per_layer,
    }
    modules = dict(tables)
    for index, layer in enumerate(model.layers):
        for name in ("up", "gate", "down"):
            modules[f"expert.{index}.{name}"] = getattr(layer.experts.switch_glu, f"{name}_proj")
    mx.save_safetensors(
        str(tmp_path / "weights.safetensors"),
        {
            f"{name}.{component}": getattr(module, component)
            for name, module in modules.items()
            for component in components
        },
    )
    catalog = TensorCatalog.inspect(tmp_path)
    logical = logical_tensors(catalog, declaration=None)
    encoding = AffineEncoding(4, 32)
    reader = PositionalReader(workers=2)
    embeddings = {
        name: StreamedEmbedding(
            AffineRowTable((tuple(catalog.tensors[f"{name}.{c}"] for c in components),), encoding),
            reader,
            budget,
            cache_bytes=65536,
            owner=f"gemma.{name}",
        )
        for name in tables
    }
    sources = tuple(
        ExpertSource(
            *(
                ProjectionSource(*(logical[f"expert.{index}.{name}.{c}"] for c in components))
                for name in ("up", "gate", "down")
            ),
            encoding,
        )
        for index in range(4)
    )
    banks = tuple(
        ExpertBank(s, 2, budget, owner=f"gemma.experts.{i}") for i, s in enumerate(sources)
    )
    scratch = ExpertBank(sources[0], 8, budget, owner="gemma.prefill")
    operations = {
        i: StreamedExperts(
            source,
            GatedExpertMath(lambda up, gate: nn.gelu_approx(gate) * up),
            bank=banks[i],
            scratch=scratch,
            reader=reader,
        )
        for i, source in enumerate(sources)
    }
    binding = bind_gemma4(
        model,
        embedding=embeddings["tokens"],
        per_layer_embedding=embeddings["per_layer"],
        experts=operations,
        attention=MetalPagedAttention(),
    )
    streamed = ModelRuntime(binding.program, resident.states, resident.owner)
    resident_row, streamed_row = resident.create(), streamed.create()
    for tokens, accepted in [((1, 2, 3, 4, 5, 6, 7, 8, 9), 9), ((3, 4, 5), 1), ((3,), 1)]:
        expected = resident.forward(resident_row, tokens)
        expected.complete()
        actual = streamed.forward(streamed_row, tokens)
        actual.complete()
        assert mx.array_equal(actual.output.logits, expected.output.logits).item()
        expected.accept(accepted)
        actual.accept(accepted)
    for operation in embeddings.values():
        assert operation.metrics["bytes_read"] > 0 and operation.metrics["cache_hits"] > 0
    resident_row.close()
    streamed_row.close()
    resident.owner.close()
    for resource in (*embeddings.values(), *banks, scratch, reader, arena):
        resource.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("shared", [False, True])
def test_compiled_decode_keeps_shared_producers_across_tail_rollover_and_rejection(bits, shared):
    model, runtime, arena, budget = compose_gemma(
        bits=bits, shared=shared, attention=MetalPagedAttention(), global_width=512
    )
    assert runtime.program.decode is not None
    prefixes = [tuple(range(1, 4)), tuple(range(1, 20))]
    rows = tuple(runtime.create() for _ in prefixes)
    for row, prefix in zip(rows, prefixes, strict=True):
        runtime.prefill(row, prefix)
    request = ForwardRequest(features=frozenset({"residual:4"}))
    for index in range(20):
        tokens = tuple(21 + (index + row) % 30 for row in range(2))
        advances = runtime.forward_batch(
            rows, tuple(ModelInputs.from_tokens((token,)) for token in tokens), request
        )
        for row_index, (advance, token) in enumerate(zip(advances, tokens, strict=True)):
            advance.complete()
            prefix = prefixes[row_index]
            expected = model(mx.array([[*prefix, token]])).logits[:, -1:]
            assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
            assert advance.output.features["residual:4"].shape == (1, 1, 64)
            accepted = 0 if row_index == 1 and index % 3 == 0 else 1
            advance.accept(accepted)
            prefixes[row_index] = (*prefix, token) if accepted else prefix
        assert rows[0].state.tail is not None
        assert all(
            row.state.length == len(prefix) for row, prefix in zip(rows, prefixes, strict=True)
        )
    assert arena.counters["tail_appended_tokens"] == 20 * 2 * len(arena.layers)
    # A fork seals its parent's live tail. A later wide forward must seal the
    # other row too, before ordinary page writes become authoritative again.
    checkpoint = rows[0].checkpoint()
    fork = runtime.create(checkpoint)
    for row, prefix in zip((*rows, fork), (*prefixes, prefixes[0]), strict=True):
        advance = runtime.forward(row, (51, 52))
        advance.complete()
        expected = model(mx.array([[*prefix, 51, 52]])).logits[:, -2:]
        assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
        advance.accept(2)
        assert row.state.tail is None
        row.close()
    checkpoint.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_compiled_feature_only_decode_completes_producer_state():
    _, runtime, arena, budget = compose_gemma(bits=4, shared=True, attention=MetalPagedAttention())
    row = runtime.create()
    runtime.prefill(row, (1, 2, 3))
    advance = runtime.forward(row, (4,), ForwardRequest(logits=False))
    advance.complete()
    advance.accept(1)
    assert advance.output.logits is None
    assert row.state.tail is not None
    expected = tuple(row.state.read(i) for i in range(len(arena.layers)))
    checkpoint = row.checkpoint()
    restored = runtime.create(checkpoint)
    for i, (k, v) in enumerate(expected):
        actual_k, actual_v = restored.state.read(i)
        assert mx.array_equal(actual_k, k).item()
        assert mx.array_equal(actual_v, v).item()
    restored.close()
    checkpoint.close()
    row.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_text_loading_keeps_declared_media_out_of_parameter_materialization(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict

    from mlx.utils import tree_flatten

    from magnitude_engine.models.architectures.gemma4.loading import load_gemma4
    from magnitude_engine.models.embeddings.binding import Resident as EmbeddingFactory
    from magnitude_engine.models.experts.binding import Resident as ExpertFactory
    from magnitude_engine.resources.io.reader import PositionalReader

    model, runtime, arena, _ = compose_gemma(bits=4, shared=True, attention=GatheredAttention())
    runtime.owner.close()
    arena.close()
    parameters = {"language_model." + k: v for k, v in tree_flatten(model.parameters())}
    media = {f"{kind}_tower.test.weight": mx.ones((8, 8)) for kind in ("audio", "vision")}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {**parameters, **media})
    config = {"text_config": asdict(model.args), "quantization": {"bits": 4, "group_size": 32}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        "magnitude_engine.models.architectures.gemma4.loading.tokenizer_identity", lambda _: "test"
    )
    budget, reader = MemoryBudget(32 << 20), PositionalReader(workers=1)

    def load():
        return load_gemma4(
            tmp_path,
            budget=budget,
            reader=reader,
            attention=GatheredAttention(),
            embedding_factory=EmbeddingFactory(),
            per_layer_embedding_factory=EmbeddingFactory(),
            expert_factory=ExpertFactory(),
        )

    try:
        with pytest.raises(ValueError, match="declared configuration"):
            load()
        assert budget.snapshot().reserved == 0
        config.update(audio_config={"model_type": "test"}, vision_config={"model_type": "test"})
        config_path.write_text(json.dumps(config))
        loaded = load()
        assert set(loaded.media_tensors) == set(media)
        assert budget.snapshot().reserved == sum(t.nbytes for t in parameters.values())
        loaded.close()
        assert budget.snapshot().reserved == 0
    finally:
        reader.close()
