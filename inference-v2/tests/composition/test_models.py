import mlx.core as mx
import pytest

from magnitude_engine import blueprints as bp
from tests.models.architectures.qwen35.test_construction import artifact_pair


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("full_attention_interval", [1, 2])
@pytest.mark.parametrize(
    "embedding_streamed,experts_streamed",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_artifact_composition_substitutions_preserve_model_and_state(
    tmp_path, bits, full_attention_interval, embedding_streamed, experts_streamed
):
    path, _, oracle, _ = artifact_pair(
        tmp_path,
        bits=bits,
        full_attention_interval=full_attention_interval,
    )
    reader = bp.resources.io.PositionalReader(workers=1)
    embedding = (
        bp.model.embeddings.Streamed(cache_bytes=8192, reader=reader)
        if embedding_streamed
        else bp.model.embeddings.Resident()
    )
    experts = (
        bp.model.experts.Streamed(slots=2, reader=reader)
        if experts_streamed
        else bp.model.experts.Resident()
    )
    program = bp.model.programs.qwen35.Program(
        artifact=bp.model.artifacts.Local(path=str(path)),
        reader=reader,
        embedding=embedding,
        feedforward=bp.model.feedforward.qwen35.MoE(experts=experts),
    )
    declaration = bp.engine.Engine(
        generation=bp.generation.Generation(
            target=bp.model.Executor(
                program=program,
                state=bp.model.state.PagedHybrid(page_size=4, slab_pages=4),
            )
        ),
        memory=bp.engine.memory.Budgeted(limit_bytes=64 << 20),
        context_tokens=128,
    )
    with bp.build(bp.loads(bp.dumps(declaration))) as engine:
        model = engine.engine.generation.model
        budget = engine.budget
        row = model.create()
        model.prefill(row, (1, 2, 3))
        checkpoint = row.checkpoint()
        branch = model.create(checkpoint)
        for sequence, tokens, accepted in ((row, (4, 5, 6), 1), (branch, (7, 8), 2)):
            advance = model.forward(sequence, tokens)
            advance.complete()
            expected = oracle(mx.array([[1, 2, 3, *tokens]]))[:, -len(tokens) :]
            assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
            advance.accept(accepted)
            sequence.close()
        checkpoint.close()
    assert budget.snapshot().reserved == 0


def test_program_state_incompatibility_fails_before_artifact_access():
    source = bp.model.upstream.mlx_vlm.ModelLoader(
        artifact=bp.model.artifacts.Local(path="/does/not/exist")
    )
    target = bp.model.Executor(
        program=bp.model.programs.mlx_vlm.Forward(source=source), state=bp.model.state.PagedHybrid()
    )
    with pytest.raises(ValueError, match="native program"), bp.build(target):
        pytest.fail("incompatible state was accepted")


@pytest.mark.parametrize("implementation", ["upstream", "resident", "streamed"])
def test_gemma_artifact_composition_preserves_shared_kv_and_per_layer_inputs(
    tmp_path, implementation
):
    import json
    from dataclasses import asdict

    from mlx.utils import tree_flatten

    from magnitude_engine.models.attention.gathered import GatheredAttention
    from tests.models.architectures.gemma4.test_gemma_model import compose_gemma

    oracle, reference, arena, budget = compose_gemma(
        bits=4,
        shared=True,
        attention=GatheredAttention(),
    )
    reference.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "text_config": asdict(oracle.args),
                "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_text("{}")
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {"language_model." + name: value for name, value in tree_flatten(oracle.parameters())},
    )
    artifact = bp.model.artifacts.Local(path=str(tmp_path))
    if implementation == "upstream":
        target = bp.model.auto(artifact)
    else:
        embedding = (
            bp.model.embeddings.Streamed(cache_bytes=8192)
            if implementation == "streamed"
            else bp.model.embeddings.Resident()
        )
        experts = (
            bp.model.experts.Streamed(slots=2)
            if implementation == "streamed"
            else bp.model.experts.Resident()
        )
        target = bp.model.Executor(
            program=bp.model.programs.gemma4.Program(
                artifact=artifact,
                embedding=embedding,
                per_layer_embedding=embedding,
                experts=experts,
            ),
            state=bp.model.state.PagedHybrid(page_size=4, slab_pages=4),
        )
    engine_bp = bp.engine.Engine(
        generation=bp.generation.Generation(target=target),
        memory=bp.engine.memory.Budgeted(limit_bytes=64 << 20),
        context_tokens=128,
    )
    with bp.build(bp.loads(bp.dumps(engine_bp))) as engine:
        model = engine.engine.generation.model
        actual_budget = engine.budget
        row = model.create()
        model.prefill(row, tuple(range(1, 12)))
        checkpoint = row.checkpoint()
        branch = model.create(checkpoint)
        for sequence, tokens, accepted in ((row, (21, 22, 23), 1), (branch, (31, 32), 2)):
            advance = model.forward(sequence, tokens)
            advance.complete()
            expected = oracle(mx.array([[*range(1, 12), *tokens]])).logits[:, -len(tokens) :]
            assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
            advance.accept(accepted)
            sequence.close()
        checkpoint.close()
    assert actual_budget.snapshot().reserved == 0
