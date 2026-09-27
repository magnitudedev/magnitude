from dataclasses import replace

import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

from magnitude_engine.models.architectures.qwen35.attention.operation import GatedAttention
from magnitude_engine.models.architectures.qwen35.attention.rotary import QwenRotary
from magnitude_engine.models.architectures.qwen35.decode import ResidentDecode
from magnitude_engine.models.architectures.qwen35.inputs import QwenInputs
from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.embeddings.replacement import EmbeddingReplacement
from magnitude_engine.models.inputs import ModelInputs
from tests.models.architectures.qwen35.test_hybrid_model import setup


def test_multiaxis_rotary_matches_independent_equation_and_preserves_unrotated_tail():
    mx.random.seed(100)
    source = mx.random.normal((2, 4, 5, 32))
    positions = mx.array(
        [
            [[1, 2, 2, 2, 3], [5, 6, 6, 6, 7]],
            [[1, 2, 3, 4, 5], [5, 6, 8, 10, 12]],
            [[1, 4, 5, 6, 7], [5, 7, 9, 11, 13]],
        ],
        mx.int32,
    )
    rotary = QwenRotary(Qwen3_5RotaryEmbedding(16, base=10000, mrope_section=[3, 3, 2]))
    actual, _ = rotary(source, source[:, :1], offset=positions)
    # Qwen interleaves height/width frequencies, leaving the remaining
    # frequencies on the temporal axis; pairs span the two rotary halves.
    axes = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    coordinates = np.array(positions)[axes].transpose(1, 2, 0)
    frequencies = 10000.0 ** (-np.arange(0, 16, 2, dtype=np.float64) / 16)
    angles = coordinates[:, None] * frequencies
    x = np.array(source).astype(np.float64)
    expected = x.copy()
    expected[..., :8] = x[..., :8] * np.cos(angles) - x[..., 8:16] * np.sin(angles)
    expected[..., 8:16] = x[..., 8:16] * np.cos(angles) + x[..., :8] * np.sin(angles)
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-6, rtol=2e-6)
    assert np.array_equal(np.array(actual)[..., 16:], x[..., 16:])


def test_compiled_continuation_keeps_rotary_offsets_separate_from_history_positions():
    _, runtime, arena, budget = setup(attention=MetalPagedAttention(), head_width=32)
    program = runtime.program
    program.blocks = tuple(
        replace(
            block,
            mixer=replace(
                block.mixer,
                positions=QwenRotary(
                    Qwen3_5RotaryEmbedding(8, base=100000, mrope_section=[2, 1, 1])
                ),
            )
            if isinstance(block.mixer, GatedAttention)
            else block.mixer,
        )
        for block in program.blocks
    )
    program.decode = ResidentDecode(program)
    first, second = runtime.create(), runtime.create()
    coordinates = mx.array([[[0, 1, 1, 2]], [[0, 1, 2, 3]], [[0, 2, 3, 4]]], mx.int32)
    values = mx.random.normal((1, 2, 64))
    inputs = ModelInputs(
        mx.array([[1, 2, 2, 3]], mx.int32),
        data=QwenInputs(coordinates, (EmbeddingReplacement(1, values),)),
    )
    runtime.prefill(first, inputs)
    runtime.prefill(second, inputs)
    step = ModelInputs(mx.array([[4]], mx.int32), data=QwenInputs(mx.array([7], mx.int32)))
    compiled = runtime.forward(first, step)
    compiled.accept(1)
    program.decode = None
    scoped = runtime.forward(second, step)
    scoped.accept(1)
    assert first.position == second.position == 5
    assert mx.allclose(compiled.output.logits, scoped.output.logits, atol=2e-5).item()
    first.close()
    second.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_projected_input_and_restored_continuation_match_upstream_decoder():
    from dataclasses import asdict

    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    source, runtime, arena, budget = setup(attention=MetalPagedAttention(), head_width=32)
    config = TextConfig.from_dict(
        {
            **asdict(source.args),
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 100000.0,
                "mrope_section": [2, 1, 1],
                "rope_type": "default",
            },
        }
    )
    reference = LanguageModel(config)
    reference.load_weights(tree_flatten(source.parameters()))
    reference.eval()
    runtime.program.blocks = tuple(
        replace(
            block,
            mixer=replace(
                block.mixer, positions=QwenRotary(reference.layers[index].self_attn.rotary_emb)
            ),
        )
        if isinstance(block.mixer, GatedAttention)
        else block
        for index, block in enumerate(runtime.program.blocks)
    )
    runtime.program.decode = ResidentDecode(runtime.program)
    tokens = mx.array([[1, 2, 2, 3]], mx.int32)
    coordinates = mx.array([[[0, 1, 1, 3]], [[0, 1, 2, 3]], [[0, 2, 1, 3]]], mx.int32)
    projected = mx.random.normal((1, 2, 64), key=mx.random.key(21))
    ordinary = reference.model.embed_tokens(tokens)
    embeddings = mx.concatenate([ordinary[:, :1], projected, ordinary[:, 3:]], axis=1)
    cache = reference.make_cache()
    hidden = reference.model(
        tokens, inputs_embeds=embeddings, position_ids=coordinates, cache=cache
    )
    expected = reference.model.embed_tokens.as_linear(hidden)
    row = runtime.create()
    checkpoint = None
    for start in (0, 2):
        data = QwenInputs(
            coordinates[:, :, start : start + 2],
            (
                EmbeddingReplacement(
                    1 if start == 0 else 0, projected[:, :1] if start == 0 else projected[:, 1:]
                ),
            ),
        )
        actual = runtime.forward(row, ModelInputs(tokens[:, start : start + 2], data=data))
        actual.accept(2)
        assert mx.allclose(
            actual.output.logits, expected[:, start : start + 2], atol=2e-4, rtol=2e-4
        ).item()
        if start == 0:
            checkpoint = row.checkpoint()
    assert checkpoint is not None
    restored = runtime.create(checkpoint)
    tail = ModelInputs(
        tokens[:, 2:],
        data=QwenInputs(coordinates[:, :, 2:], (EmbeddingReplacement(0, projected[:, 1:]),)),
    )
    actual = runtime.forward(restored, tail)
    actual.accept(2)
    assert mx.allclose(actual.output.logits, expected[:, 2:], atol=2e-4, rtol=2e-4).item()
    next_token = mx.array([[4]], mx.int32)
    hidden = reference.model(next_token, cache=cache, position_ids=mx.array([[5]], mx.int32))
    expected_next = reference.model.embed_tokens.as_linear(hidden)
    for current in (row, restored):
        actual = runtime.forward(
            current, ModelInputs(next_token, data=QwenInputs(mx.array([5], mx.int32)))
        )
        actual.accept(1)
        assert mx.allclose(actual.output.logits, expected_next, atol=2e-4, rtol=2e-4).item()
        current.close()
    checkpoint.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0
