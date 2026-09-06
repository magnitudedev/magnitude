import json
from dataclasses import asdict, replace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from magnitude_engine.generation.methods.mtp.runtime import MTPMethod
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.architectures.qwen35.attention.binding import Attention
from magnitude_engine.models.architectures.qwen35.feedforward.binding import MoE
from magnitude_engine.models.architectures.qwen35.loading import load_qwen35
from magnitude_engine.models.architectures.qwen35.mtp.loading import MTPParameters, load_mtp
from magnitude_engine.models.architectures.qwen35.recurrence.binding import Mixer
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.embeddings.binding import Resident as ResidentEmbeddingFactory
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.experts.binding import Resident as ResidentExpertFactory
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.recurrence.metal import MetalDelta
from magnitude_engine.models.runtime import ForwardRequest, ModelRuntime
from magnitude_engine.models.state.arena import KVArena
from magnitude_engine.models.state.hybrid import HybridStateStore
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader


def artifact_pair(directory, bits=4, *, full_attention_interval=2):
    mx.random.seed(7)
    args = TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=128,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        full_attention_interval=full_attention_interval,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
    )
    model = TextModel(args)
    head = MTPParameters(replace(args, num_hidden_layers=1, full_attention_interval=1))
    # A non-unit norm makes any accidental base-model +1 sanitization observable.
    head.norm.weight = mx.linspace(0.2, 1.2, args.hidden_size)
    settings = {"bits": bits, "group_size": 64, "mode": "affine"}

    def policy(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith(("mlp.gate", "shared_expert_gate")):
            value = {"bits": 8, "group_size": 64, "mode": "affine"}
            settings["language_model." + path] = value
            return value
        return True

    nn.quantize(model, bits=bits, group_size=64, class_predicate=policy)
    target_path, head_path = directory / "target", directory / "head"
    target_path.mkdir()
    head_path.mkdir()
    (target_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": asdict(args),
                "quantization": settings,
            }
        )
    )
    (target_path / "tokenizer_config.json").write_text('{"test_vocabulary": "sequential"}')
    (head_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {**asdict(args), "mtp_num_hidden_layers": 1},
                "block_size": 3,
            }
        )
    )
    mx.save_safetensors(
        str(target_path / "model.safetensors"),
        {"language_model." + name: value for name, value in tree_flatten(model.parameters())},
    )
    mx.save_safetensors(
        str(head_path / "model.safetensors"),
        {"mtp." + name: value for name, value in tree_flatten(head.parameters())},
    )
    nn.quantize(head, bits=bits, group_size=64, class_predicate=policy)
    model.eval()
    head.eval()
    return target_path, head_path, model, head


@pytest.mark.parametrize("bits", [4, 8])
def test_loaded_target_and_paired_head_numerics_generation_and_lifetimes(tmp_path, bits):
    target_path, head_path, reference, reference_head = artifact_pair(tmp_path, bits)
    budget, reader, owner = MemoryBudget(32 << 20), PositionalReader(workers=2), ExecutionOwner()
    loaded = load_qwen35(
        target_path,
        budget=budget,
        reader=reader,
        attention=Attention(GatheredAttention()),
        recurrence=Mixer(MetalDelta()),
        feedforward=MoE(ResidentExpertFactory()),
        embedding_factory=ResidentEmbeddingFactory(),
    )
    head = load_mtp(head_path, loaded, budget=budget, reader=reader)
    assert head.capacity == 2 and head.depth == 1
    with pytest.raises(RuntimeError, match="borrowed"):
        loaded.close()
    arena = KVArena(
        loaded.attention,
        page_size=4,
        slab_pages=4,
        max_pages=64,
        budget=budget,
        dtype=loaded.state_dtype,
    )
    target = ModelRuntime(
        loaded.program, HybridStateStore(PageStore(arena), loaded.recurrence, budget), owner
    )
    drafter = ModelRuntime(head.program, head.state_store(budget), owner)
    row = target.create()
    advance = target.forward(row, (1, 2, 3), ForwardRequest(True, frozenset({"residual:2"})))
    with pytest.raises(RuntimeError, match="active"):
        loaded.close()
    advance.complete()
    expected = reference(mx.array([[1, 2, 3]]), cache=reference.make_cache())
    assert mx.allclose(advance.output.logits, expected, atol=1e-5).item()
    inputs = ModelInputs(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        {
            "previous_hidden": advance.output.features["residual:2"],
        },
    )
    private_row = drafter.create()
    prediction = drafter.forward(private_row, inputs, ForwardRequest(True))
    with pytest.raises(RuntimeError, match="active"):
        head.close()
    embedded = reference.model.embed_tokens(inputs.tokens)
    hidden = reference_head.fc(
        mx.concatenate(
            [
                reference_head.pre_fc_norm_embedding(embedded),
                reference_head.pre_fc_norm_hidden(inputs.conditioning["previous_hidden"]),
            ],
            axis=-1,
        )
    )
    for layer in reference_head.layers:
        hidden = layer(hidden, mask="causal", cache=KVCache())
    expected = reference.lm_head(reference_head.norm(hidden))
    prediction.complete()
    assert mx.allclose(prediction.output.logits, expected, atol=1e-5).item()
    prediction.accept(2)
    private_row.close()
    row.close()
    method = MTPMethod(
        target=target,
        head=drafter,
        target_feature=head.target_feature,
        project=head.vocabulary.project,
        capacity=head.capacity,
        budget=budget,
        identity="fixture",
    )
    idle = method.create(target=target)
    with pytest.raises(RuntimeError, match="active sequences"):
        head.close()
    idle.close()
    outputs = []
    for strategy in (method, PlainMethod()):
        sequence = GenerationRuntime(target, strategy).create(
            (1, 2, 3),
            SamplingPolicy(temperature=0),
            12,
        )
        result = []
        while not sequence.finished:
            result.extend(sequence.step(3).tokens)
        outputs.append(result)
        sequence.close()
    assert outputs[0] == outputs[1]
    owner.close()
    arena.close()
    head.close()
    reserved = budget.snapshot().reserved
    with pytest.raises(RuntimeError, match="closed"):
        # Use an available execution owner so this exercises the program lifetime.
        ModelRuntime(head.program, head.state_store(budget), ExecutionOwner()).create()
    assert budget.snapshot().reserved == reserved
    loaded.close()
    reader.close()
    assert budget.snapshot().reserved == 0


def test_head_preflight_and_allocation_failure_do_not_leave_target_borrowed(tmp_path):
    target_path, head_path, _, _ = artifact_pair(tmp_path)
    budget, reader = MemoryBudget(32 << 20), PositionalReader(workers=1)
    target = load_qwen35(
        target_path,
        budget=budget,
        reader=reader,
        attention=Attention(GatheredAttention()),
        recurrence=Mixer(MetalDelta()),
        feedforward=MoE(ResidentExpertFactory()),
        embedding_factory=ResidentEmbeddingFactory(),
    )
    before = budget.snapshot().reserved
    budget.limit = before
    with pytest.raises(MemoryError):
        load_mtp(head_path, target, budget=budget, reader=reader)
    assert budget.snapshot().reserved == before
    config = json.loads((head_path / "config.json").read_text())
    config["text_config"]["hidden_size"] = 128
    (head_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="geometry"):
        load_mtp(head_path, target, budget=budget, reader=reader)
    target.close()
    reader.close()
    assert budget.snapshot().reserved == 0


def test_text_construction_preserves_declared_vision_records_and_rejects_unknown_weights(tmp_path):
    target_path, _, _, _ = artifact_pair(tmp_path)
    path = target_path / "model.safetensors"
    tensors = mx.load(str(path))
    mx.eval(*tensors.values())
    tensors["vision_tower.test.weight"] = mx.ones((8, 8))
    mx.save_safetensors(str(path), tensors)
    config_path = target_path / "config.json"
    config = json.loads(config_path.read_text())
    budget, reader = MemoryBudget(32 << 20), PositionalReader(workers=1)

    def load():
        return load_qwen35(
            target_path,
            budget=budget,
            reader=reader,
            attention=Attention(GatheredAttention()),
            recurrence=Mixer(MetalDelta()),
            feedforward=MoE(ResidentExpertFactory()),
            embedding_factory=ResidentEmbeddingFactory(),
        )

    with pytest.raises(ValueError, match="declared"):
        load()
    assert budget.snapshot().reserved == 0
    config["vision_config"] = {"model_type": "test"}
    config_path.write_text(json.dumps(config))
    target = load()
    assert set(target.vision_tensors) == {"vision_tower.test.weight"}
    assert budget.snapshot().reserved == sum(
        a.nbytes for name, a in tensors.items() if not name.startswith("vision_tower.")
    )
    target.close()
    tensors["unknown.weight"] = mx.ones((8, 8))
    mx.save_safetensors(str(path), tensors)
    with pytest.raises(ValueError, match="layout differs"):
        load()
    assert budget.snapshot().reserved == 0
    reader.close()
