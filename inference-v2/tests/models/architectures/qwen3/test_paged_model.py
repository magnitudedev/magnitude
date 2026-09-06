import mlx.core as mx
import pytest
from mlx_lm.models.qwen3 import Model, ModelArgs

from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.embeddings.resident import ResidentEmbedding
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.runtime import ForwardRequest, ModelRuntime
from magnitude_engine.models.state.arena import KVArena, LayerGeometry
from magnitude_engine.models.state.paged import PagedStateStore
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.resources.budget import MemoryBudget

from .fixtures import Qwen3Program, QwenAttention, QwenBlock


def compose(model):
    blocks = []
    for layer in model.layers:
        attention = layer.self_attn
        blocks.append(
            QwenBlock(
                layer.input_layernorm,
                QwenAttention(
                    attention.q_proj,
                    attention.k_proj,
                    attention.v_proj,
                    attention.o_proj,
                    attention.q_norm,
                    attention.k_norm,
                    attention.rope,
                    attention.n_heads,
                    attention.n_kv_heads,
                    model.args.head_dim,
                ),
                layer.post_attention_layernorm,
                layer.mlp,
            )
        )
    return Qwen3Program(
        ResidentEmbedding(model.model.embed_tokens.weight),
        tuple(blocks),
        model.model.norm,
        model.model.embed_tokens.as_linear,
        GatheredAttention(),
    )


@pytest.mark.parametrize("head_width", [8, 16])
def test_paged_neural_program_matches_library_across_branches_and_rollback(head_width):
    mx.random.seed(73)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            rope_theta=10000,
            head_dim=head_width,
            tie_word_embeddings=True,
        )
    )
    budget = MemoryBudget(1 << 20)
    arena = KVArena(
        (LayerGeometry(2, head_width, head_width),) * 2,
        page_size=4,
        slab_pages=4,
        max_pages=32,
        budget=budget,
        dtype=mx.float32,
    )
    pages = PageStore(arena)
    runtime = ModelRuntime(compose(model), PagedStateStore(pages), ExecutionOwner())
    row = runtime.create()
    runtime.prefill(row, (1, 2, 3))
    checkpoint = row.checkpoint()
    branch = runtime.create(checkpoint)
    assert arena.counters["partial_page_copies"] == 1
    step = runtime.forward(row, (4, 5, 6, 7), ForwardRequest(features=frozenset({"residual:2"})))
    step.complete()
    expected = model(mx.array([[1, 2, 3, 4, 5, 6, 7]]))[:, -4:]
    assert mx.allclose(step.output.logits, expected, atol=1e-5).item()
    assert step.output.features["residual:2"].shape == (1, 4, 32)
    step.accept(2)
    assert row.state.length == 5
    continuation = runtime.forward(row, (8, 9))
    continuation.complete()
    expected = model(mx.array([[1, 2, 3, 4, 5, 8, 9]]))[:, -2:]
    assert mx.allclose(continuation.output.logits, expected, atol=1e-5).item()
    continuation.accept(2)
    independent = runtime.forward(branch, (9, 8))
    independent.complete()
    expected = model(mx.array([[1, 2, 3, 9, 8]]))[:, -2:]
    assert mx.allclose(independent.output.logits, expected, atol=1e-5).item()
    independent.accept(2)
    pages.validate()
    checkpoint.close()
    branch.close()
    row.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_paged_prefill_skips_output_projection_and_pins_pending_consumers():
    mx.random.seed(4)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=1,
            max_position_embeddings=1024,
            rope_theta=10000,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )
    program = compose(model)
    calls = []
    output = program.output

    def project(value):
        calls.append(value.shape)
        return output(value)

    program.output = project
    arena = KVArena(
        (LayerGeometry(1, 8, 8),),
        page_size=4,
        slab_pages=4,
        max_pages=16,
        budget=MemoryBudget(1 << 20),
        dtype=mx.float32,
    )
    runtime = ModelRuntime(program, PagedStateStore(PageStore(arena)), ExecutionOwner())
    row = runtime.create()
    runtime.prefill(row, (1, 2, 3))
    assert not calls
    pending = runtime.forward(row, (4,))
    with pytest.raises(RuntimeError, match="pin"):
        arena.shrink()
    pending.accept(1)
    assert calls == [(1, 1, 16)]
    row.close()
    arena.close()
