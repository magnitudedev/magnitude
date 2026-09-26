import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.kernels.contractions.weights import ExpertWeights, QuantizedProjection
from magnitude_engine.models.architectures.qwen35.attention.binding import Attention
from magnitude_engine.models.architectures.qwen35.binding import bind_qwen35
from magnitude_engine.models.architectures.qwen35.feedforward.binding import MoE
from magnitude_engine.models.architectures.qwen35.recurrence.binding import Mixer
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding, ResidentEmbedding
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.experts.binding import Resident as ResidentExpertFactory
from magnitude_engine.models.experts.computation import GatedExpertMath, ResidentExperts
from magnitude_engine.models.recurrence.metal import MetalDelta
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.arena import KVArena
from magnitude_engine.models.state.hybrid import HybridStateStore
from magnitude_engine.models.state.pages import PageStore
from magnitude_engine.resources.budget import MemoryBudget
from tests.models.architectures.qwen35.library import vision_language_parameters


def setup(moe=False, bits=None, *, attention=None, head_width=16, dtype=mx.float32):
    mx.random.seed(13)
    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_width,
        vocab_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        tie_word_embeddings=True,
        num_experts=8 if moe else 0,
        num_experts_per_tok=2 if moe else 0,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )
    model = TextModel(args)
    model.eval()
    if bits is not None:
        nn.quantize(model, bits=bits, group_size=32)
    model.set_dtype(dtype)
    emb = model.model.embed_tokens
    embedding = (
        ResidentEmbedding(emb.weight)
        if bits is None
        else ResidentAffineEmbedding(emb.weight, emb.scales, emb.biases, AffineEncoding(bits, 32))
    )
    experts = {}
    if moe:
        for i, layer in enumerate(model.layers):

            def projection(module):
                return QuantizedProjection(
                    module.weight,
                    module.scales,
                    module.biases,
                    AffineEncoding(module.bits, module.group_size),
                )

            m = layer.mlp.switch_mlp
            experts[i] = ResidentExperts(
                ExpertWeights(
                    projection(m.up_proj), projection(m.gate_proj), projection(m.down_proj)
                ),
                GatedExpertMath(lambda up, gate: nn.silu(gate) * up),
            )
    binding = bind_qwen35(
        vision_language_parameters(model),
        embedding=embedding,
        experts=experts,
        attention=Attention(attention if attention is not None else GatheredAttention()),
        recurrence=Mixer(MetalDelta()),
        feedforward=MoE(ResidentExpertFactory()),
        state_dtype=dtype,
    )
    budget = MemoryBudget(8 << 20)
    arena = KVArena(
        binding.attention, page_size=4, slab_pages=4, max_pages=64, budget=budget, dtype=dtype
    )
    states = HybridStateStore(PageStore(arena), binding.recurrence, budget)
    runtime = ModelRuntime(binding.program, states, ExecutionOwner())
    return model, runtime, arena, budget


@pytest.mark.parametrize("accepted", [0, 1, 3, 4])
@pytest.mark.parametrize("moe,bits", [(False, None), (True, 4), (True, 8)])
@pytest.mark.parametrize(
    "attention",
    [GatheredAttention(), MetalPagedAttention(), MetalPagedAttention(heads_per_group=2)],
)
def test_hybrid_reconciliation_matches_library_without_transformer_replay(
    accepted, moe, bits, attention
):
    model, runtime, arena, budget = setup(moe, bits, attention=attention, head_width=32)
    row = runtime.create()
    runtime.prefill(row, (1, 2, 3))
    checkpoint = row.checkpoint()
    block = (4, 5, 6, 7)
    advance = runtime.forward(row, block)
    advance.complete()
    library_cache = model.make_cache()
    expected = model(mx.array([[1, 2, 3, *block]]), cache=library_cache)[:, -4:]
    assert mx.allclose(advance.output.logits, expected, atol=1e-4).item()
    program = runtime.program
    original_forward = program.forward

    def no_forward(*args, **kwargs):
        raise AssertionError("hybrid reconciliation must not replay transformer execution")

    program.forward = no_forward
    advance.accept(accepted)
    program.forward = original_forward
    assert row.state.position == 3 + accepted
    next_step = runtime.forward(row, (8, 9))
    next_step.complete()
    expected_cache = model.make_cache()
    expected = model(mx.array([[1, 2, 3, *block[:accepted], 8, 9]]), cache=expected_cache)[:, -2:]
    assert mx.allclose(next_step.output.logits, expected, atol=1e-4).item()
    next_step.accept(2)
    recurrent = [
        c for layer, c in zip(model.layers, expected_cache, strict=True) if layer.is_linear
    ]
    for slot, cache in zip(row.state.slots, recurrent, strict=True):
        for actual, expected in zip(slot.values, cache.state, strict=True):
            assert mx.allclose(actual, expected, atol=1e-4).item()
    branch = runtime.create(checkpoint)
    branched = runtime.forward(branch, (9,))
    branched.complete()
    expected = model(mx.array([[1, 2, 3, 9]]), cache=model.make_cache())[:, -1:]
    assert mx.allclose(branched.output.logits, expected, atol=1e-4).item()
    branch.close()
    checkpoint.close()
    row.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_hybrid_reservation_failure_keeps_committed_state_usable():
    _, runtime, arena, budget = setup()
    row = runtime.create()
    runtime.prefill(row, (1, 2, 3))
    previous = tuple(slot.values for slot in row.state.slots)
    budget.limit = budget.snapshot().reserved
    with pytest.raises(MemoryError):
        runtime.forward(row, (4, 5))
    assert row.state.position == 3 and not row.state.active
    assert all(
        slot.values is values for slot, values in zip(row.state.slots, previous, strict=True)
    )
    budget.limit = 8 << 20
    runtime.prefill(row, (4,))
    row.close()
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("conditioned", [False, True])
def test_full_hybrid_program_swaps_streamed_experts_without_library_patches(tmp_path, conditioned):
    from magnitude_engine.artifacts.layouts import logical_tensors
    from magnitude_engine.artifacts.tensors import TensorCatalog
    from magnitude_engine.models.architectures.qwen35.inputs import QwenInputs
    from magnitude_engine.models.embeddings.replacement import EmbeddingReplacement
    from magnitude_engine.models.experts.bank import ExpertBank, ExpertSource, ProjectionSource
    from magnitude_engine.models.experts.streaming import StreamedExperts
    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.resources.io.reader import PositionalReader

    model, resident, arena, budget = setup(moe=True, bits=4)
    arrays = {}
    layer_types = tuple(type(layer) for layer in model.layers)
    for index, layer in enumerate(model.layers):
        for name in ("up", "gate", "down"):
            projection = getattr(layer.mlp.switch_mlp, f"{name}_proj")
            for component in ("weight", "scales", "biases"):
                arrays[f"{index}.{name}.{component}"] = getattr(projection, component)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), arrays)
    tensors = logical_tensors(TensorCatalog.inspect(tmp_path), declaration=None)
    sources = []
    for index in range(len(model.layers)):
        projections = [
            ProjectionSource(
                *(
                    tensors[f"{index}.{name}.{component}"]
                    for component in ("weight", "scales", "biases")
                )
            )
            for name in ("up", "gate", "down")
        ]
        sources.append(ExpertSource(*projections, AffineEncoding(4, 32)))
    reader = PositionalReader()
    banks = [
        ExpertBank(source, 2, budget, owner=f"experts:{index}")
        for index, source in enumerate(sources)
    ]
    scratch = ExpertBank(sources[0], 8, budget, owner="shared-prefill-scratch")
    operations = {
        i: StreamedExperts(
            source,
            GatedExpertMath(lambda up, gate: nn.silu(gate) * up),
            bank=banks[i],
            scratch=scratch,
            reader=reader,
        )
        for i, source in enumerate(sources)
    }
    binding = bind_qwen35(
        vision_language_parameters(model),
        embedding=resident.program.embedding,
        experts=operations,
        attention=Attention(GatheredAttention()),
        recurrence=Mixer(MetalDelta()),
        feedforward=MoE(ResidentExpertFactory()),
        state_dtype=mx.float32,
    )
    streamed = ModelRuntime(binding.program, resident.states, resident.owner)
    expected = resident.create()
    actual = streamed.create()
    prompt = (1, 2, 3)
    if conditioned:
        prompt = ModelInputs(
            mx.array([prompt], mx.int32),
            data=QwenInputs(
                mx.array([0], mx.int32), (EmbeddingReplacement(1, mx.random.normal((1, 1, 64))),)
            ),
        )
    for runtime, row in ((resident, expected), (streamed, actual)):
        runtime.prefill(row, prompt)
        verify = runtime.forward(row, (4, 5, 6))
        verify.accept(1)
    resident_step = resident.forward(expected, (7,))
    resident_step.complete()
    streamed_step = streamed.forward(actual, (7,))
    streamed_step.complete()
    assert mx.array_equal(resident_step.output.logits, streamed_step.output.logits).item()
    assert tuple(type(layer) for layer in model.layers) == layer_types
    expected.close()
    actual.close()
    for bank in banks:
        bank.close()
    scratch.close()
    reader.close()
    resident.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("moe,bits", [(False, None), (True, 4), (True, 8)])
def test_physical_batch_has_independent_positions_acceptance_and_lifetimes(count, moe, bits):
    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.runtime import ForwardRequest

    model, runtime, arena, budget = setup(moe, bits, attention=MetalPagedAttention(), head_width=32)
    prefixes = ((), (1, 2, 3), (2, 4, 6, 8, 10, 12, 14), (1, 3, 5, 7, 9))
    rows = tuple(runtime.create() for _ in prefixes)
    for row, prefix in zip(rows, prefixes, strict=True):
        runtime.prefill(row, prefix)
    checkpoints = tuple(row.checkpoint() for row in rows)
    calls = []
    original = runtime.program.output

    def record_output(hidden):
        calls.append(hidden.shape)
        return original(hidden)

    runtime.program.output = record_output
    tokens = tuple(tuple(range(20 + i, 20 + i + count)) for i in range(4))
    advances = runtime.forward_batch(
        rows,
        tuple(ModelInputs.from_tokens(t) for t in tokens),
        ForwardRequest(features=frozenset({"residual:4"})),
    )
    assert calls == [(4, count, 64)]  # One physical projection, not serial forwards.
    assert len({id(advance.execution) for advance in advances}) == 1
    advances[0].complete()
    for advance, prefix, block in zip(advances, prefixes, tokens, strict=True):
        expected = model(mx.array([[*prefix, *block]]), cache=model.make_cache())[:, -count:]
        assert mx.allclose(advance.output.logits, expected, atol=1e-4).item()
        assert advance.output.features["residual:4"].shape == (1, count, 64)
    # Closing one unresolved member completes work but never commits another row.
    rows[0].close()
    assert all(row.pending is step for row, step in zip(rows[1:], advances[1:], strict=True))
    accepted = (0, 1, count)
    for row, advance, prefix, block, keep in zip(
        rows[1:], advances[1:], prefixes[1:], tokens[1:], accepted, strict=True
    ):
        advance.accept(keep)
        assert row.state.position == len(prefix) + keep
        next_step = runtime.forward(row, (31,))
        next_step.complete()
        expected = model(mx.array([[*prefix, *block[:keep], 31]]), cache=model.make_cache())[:, -1:]
        assert mx.allclose(next_step.output.logits, expected, atol=1e-4).item()
        next_step.accept(1)
        row.close()
    # Checkpoints predate the batch and survive every row's completion/disposal.
    branches = tuple(runtime.create(c) for c in checkpoints)
    steps = runtime.forward_batch(branches, tuple(ModelInputs.from_tokens((42,)) for _ in branches))
    for branch, step, prefix in zip(branches, steps, prefixes, strict=True):
        step.complete()
        expected = model(mx.array([[*prefix, 42]]), cache=model.make_cache())[:, -1:]
        assert mx.allclose(step.output.logits, expected, atol=1e-4).item()
        branch.close()
    for checkpoint in checkpoints:
        checkpoint.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_batch_reservation_failure_unwinds_earlier_rows_without_poisoning():
    from magnitude_engine.models.inputs import ModelInputs

    _, runtime, arena, budget = setup(attention=MetalPagedAttention(), head_width=32)
    rows = (runtime.create(), runtime.create())
    for row in rows:
        runtime.prefill(row, (1, 2))
    reserved = budget.snapshot().reserved
    cost = sum(s.layout.nbytes for s in rows[0].state.slots)
    budget.limit = reserved + cost  # One destination row fits; the physical batch does not.
    with pytest.raises(MemoryError):
        runtime.forward_batch(rows, (ModelInputs.from_tokens((3,)),) * 2)
    assert budget.snapshot().reserved == reserved
    assert all(not r.state.active and not r.failed and r.state.position == 2 for r in rows)
    budget.limit = 8 << 20
    steps = runtime.forward_batch(rows, (ModelInputs.from_tokens((4,)),) * 2)
    for row, step in zip(rows, steps, strict=True):
        step.accept(1)
        row.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_failed_physical_forward_drains_group_and_releases_every_transaction():
    from magnitude_engine.models.inputs import ModelInputs

    _, runtime, arena, budget = setup(attention=MetalPagedAttention(), head_width=32)
    rows = (runtime.create(), runtime.create())
    for row in rows:
        runtime.prefill(row, (1, 2, 3))
    reserved = budget.snapshot().reserved
    project = runtime.program.output

    def fail_after_state_writes(hidden):
        assert hidden.shape[0] == 2
        raise MemoryError("injected grouped projection allocation failure")

    runtime.program.output = fail_after_state_writes
    with pytest.raises(MemoryError, match="grouped projection"):
        runtime.forward_batch(rows, (ModelInputs.from_tokens((4, 5)),) * 2)
    assert all(row.failed and row.pending is None and not row.state.active for row in rows)
    assert not runtime.owner._pending
    assert budget.snapshot().reserved == reserved
    for row in rows:
        with pytest.raises(RuntimeError, match="unavailable"):
            runtime.forward(row, (6,))
        row.close()
    # Successful drainage leaves the owner available for independently created state.
    runtime.program.output = project
    fresh = runtime.create()
    runtime.prefill(fresh, (7, 8))
    fresh.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_hybrid_known_causal_prefix_cannot_be_rejected():
    from magnitude_engine.models.runtime import ForwardRequest

    model, runtime, arena, budget = setup()
    row = runtime.create()
    runtime.prefill(row, (1, 2))
    advance = runtime.forward(row, (3, 4), ForwardRequest(committed_inputs=1))
    with pytest.raises(ValueError, match="commitment"):
        advance.accept(0)
    advance.accept(1)
    follow = runtime.forward(row, (5,), ForwardRequest(committed_inputs=1))
    expected = model(mx.array([[1, 2, 3, 5]]))[:, -1:]
    assert mx.allclose(follow.output.logits, expected, atol=1e-4, rtol=1e-4).item()
    follow.accept(1)
    row.close()
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize(
    "attention",
    [GatheredAttention(), MetalPagedAttention(), MetalPagedAttention(heads_per_group=2)],
)
def test_causal_spans_reserve_pages_before_pinning_and_retain_consumed_eos(attention):
    from magnitude_engine.generation.methods.plain.runtime import PlainMethod
    from magnitude_engine.generation.runtime import GenerationRuntime
    from magnitude_engine.generation.sampling_policy import SamplingPolicy

    _, target, arena, budget = setup(True, 4, attention=attention, head_width=32)
    generation = GenerationRuntime(target, PlainMethod())
    policy = SamplingPolicy(temperature=0)
    prompt = (1, 2, 3, 4)  # Start just before a page boundary.
    oracle = generation.create(prompt, policy, 6)
    expected = []
    while not oracle.finished:
        expected.extend(oracle.step().tokens)
    oracle.close()
    # Force EOS on the first prediction, which has already been fed to lookahead.
    row = generation.create(prompt, policy, 6, (expected[0],))
    result = row.step(4)
    assert result.tokens == (expected[0],) and result.evaluated_inputs == 2
    assert row.target_position == row.model.state.position == len(prompt) + 1
    assert not target.owner._pending and not arena._pins
    checkpoint = row.checkpoint()
    continuation = tuple(row.context) + (3,)
    warm = generation.create(continuation, policy, 6, checkpoint=checkpoint)
    cold = generation.create(continuation, policy, 6)
    actual, reference = [], []
    while not warm.finished:
        actual.extend(warm.step(4).tokens)
        assert len(target.owner._pending) <= 1
        assert bool(arena._pins) == bool(target.owner._pending)
    while not cold.finished:
        reference.extend(cold.step().tokens)
    assert actual == reference
    warm.close()
    cold.close()
    checkpoint.close()
    row.close()
    target.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("moe,bits,dtype", [(False, None, mx.float32), (True, 4, mx.bfloat16)])
def test_compiled_decode_preserves_mixed_positions_rollback_and_features(moe, bits, dtype):
    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.runtime import ForwardRequest

    _, runtime, arena, budget = setup(
        moe, bits, attention=MetalPagedAttention(heads_per_group=2), head_width=32, dtype=dtype
    )
    program = runtime.program
    decoder = program.decode
    assert decoder is not None
    checkpoints = []
    for prefix in [(1, 2, 3), (1, 2, 3, 4, 5, 6, 7)]:
        row = runtime.create()
        runtime.prefill(row, prefix)
        checkpoints.append(row.checkpoint())
        row.close()

    def replay(compiled):
        program.decode = decoder if compiled else None
        rows = tuple(runtime.create(checkpoint) for checkpoint in checkpoints)
        outputs = []
        try:
            for index in range(4):
                request = ForwardRequest(
                    logits=index != 1,
                    features=frozenset(("residual:0", "residual:2", "residual:4")),
                    committed_inputs=0 if index == 0 else 1,
                )
                advances = runtime.forward_batch(
                    rows,
                    tuple(ModelInputs.from_tokens((8 + index + row,)) for row in range(2)),
                    request,
                )
                for row, advance in enumerate(advances):
                    result = advance.output
                    arrays = tuple(result.features[name] for name in sorted(result.features))
                    if result.logits is not None:
                        arrays += (result.logits,)
                    mx.eval(arrays)
                    outputs.extend(arrays)
                    # One peer rejects the first token; their positions then diverge further.
                    advance.accept(0 if index == 0 and row == 0 else 1)
                    advance.complete()
            for row in rows:
                for layer in range(len(arena.layers)):
                    outputs.extend(row.state.pages.read(layer))
                outputs.extend(a for slot in row.state.slots for a in slot.values)
            mx.eval(outputs)
            return tuple(outputs), tuple(row.state.position for row in rows)
        finally:
            for row in rows:
                row.close()

    try:
        expected, expected_positions = replay(False)
        actual, actual_positions = replay(True)
        assert actual_positions == expected_positions == (6, 11)
        for a, e in zip(actual, expected, strict=True):
            if dtype == mx.bfloat16:
                assert bool(mx.array_equal(a, e).item())
            else:
                assert bool(mx.allclose(a, e, rtol=1e-5, atol=1e-5).item())
        assert len(decoder.functions) <= 4
    finally:
        program.decode = decoder
        for checkpoint in checkpoints:
            checkpoint.close()
        arena.close()
    assert budget.snapshot().reserved == 0
