"""Full hybrid equations and transactional continuation against FP64 reference.

The fixture is a small, independently generated GGUF. It exercises the actual
artifact loader, operation provider, program and state store, not mock operations.
"""

import os
from contextlib import ExitStack

import gguf
import numpy as np
import pytest

from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.inputs.layout import InputLayout, InputSpan
from magnitude_engine.models.qwen35.artifact import inspect_dense
from magnitude_engine.models.qwen35.inputs import Feature, InputPlan, Inputs
from magnitude_engine.models.qwen35.runtime import DenseRuntime, ForwardRequest, LogitsSelection
from magnitude_engine.models.sequence import ModelRequest
from magnitude_engine.operations.factory import ResidentOperations
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.sampling import (
    Draw,
    SampledToken,
    SamplePosition,
    SampleSelector,
    SamplingSeed,
    SelectionKind,
)
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import CapacityError, DType, TensorSpec
from magnitude_engine.platform.machine import open_context


def fixture(path):
    rng = np.random.default_rng(824)
    weights = {}

    def weight(name, shape, kind="matrix"):
        scale = 0.7 / np.sqrt(shape[-1]) if len(shape) == 2 else 0.1
        values = rng.normal(0, scale, shape)
        if kind == "norm":
            values += 1
        if kind == "decay":
            values = -np.exp(values)
        values = values.astype(np.float16 if kind == "conv" else np.float32)
        weights[name] = values.astype(np.float64)
        writer.add_tensor(name, values)

    writer = gguf.GGUFWriter(path, "qwen35")
    metadata = {
        "block_count": 3,
        "full_attention_interval": 2,
        "embedding_length": 32,
        "feed_forward_length": 64,
        "context_length": 1024,
        "attention.head_count": 4,
        "attention.head_count_kv": 2,
        "attention.key_length": 16,
        "attention.value_length": 16,
        "rope.dimension_count": 8,
        "ssm.conv_kernel": 4,
        "ssm.group_count": 2,
        "ssm.time_step_rank": 4,
        "ssm.state_size": 16,
        "ssm.inner_size": 64,
    }
    for name, value in metadata.items():
        writer.add_uint32("qwen35." + name, value)
    writer.add_float32("qwen35.rope.freq_base", 10000)
    writer.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_array("qwen35.rope.dimension_sections", (2, 1, 1, 0))
    weight("token_embd.weight", (32, 32))
    weight("output_norm.weight", (32,), "norm")
    for layer in range(3):
        prefix = f"blk.{layer}."
        weight(prefix + "attn_norm.weight", (32,), "norm")
        weight(prefix + "post_attention_norm.weight", (32,), "norm")
        weight(prefix + "ffn_gate.weight", (64, 32))
        weight(prefix + "ffn_up.weight", (64, 32))
        weight(prefix + "ffn_down.weight", (32, 64))
        if layer == 1:
            weight(prefix + "attn_q.weight", (128, 32))
            for name in ("k", "v"):
                weight(prefix + f"attn_{name}.weight", (32, 32))
            for name in ("q", "k"):
                weight(prefix + f"attn_{name}_norm.weight", (16,), "norm")
            weight(prefix + "attn_output.weight", (32, 64))
        else:
            weight(prefix + "attn_qkv.weight", (128, 32))
            weight(prefix + "attn_gate.weight", (64, 32))
            for name in ("alpha", "beta"):
                weight(prefix + f"ssm_{name}.weight", (4, 32))
            weight(prefix + "ssm_conv1d.weight", (128, 4), "conv")
            weight(prefix + "ssm_a", (4,), "decay")
            weight(prefix + "ssm_dt.bias", (4,))
            weight(prefix + "ssm_norm.weight", (16,), "norm")
            weight(prefix + "ssm_out.weight", (32, 64))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return weights


def reference(weights, tokens, *, embeddings=None, coordinates=None):
    """Direct token-by-token equations without using production numerical helpers."""
    recurrent = {i: (np.zeros((128, 3)), np.zeros((4, 16, 16))) for i in (0, 2)}
    history = []
    result = []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def norm(x, scale):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * scale

    def rotary(x, position):
        out = x.copy()
        axes = position if coordinates is None else np.asarray(coordinates[position])[[0, 1, 2, 0]]
        angle = axes * (10000.0 ** (-np.arange(4) / 4)).astype(np.float32).astype(np.float64)
        out[:, :4] = x[:, :4] * np.cos(angle) - x[:, 4:8] * np.sin(angle)
        out[:, 4:8] = x[:, 4:8] * np.cos(angle) + x[:, :4] * np.sin(angle)
        return out

    for position, token in enumerate(tokens):
        x = weights["token_embd.weight"][token] if embeddings is None else embeddings[position]
        for layer in range(3):
            prefix = f"blk.{layer}."

            def w(name, prefix=prefix):
                return weights[prefix + name]

            a = norm(x, w("attn_norm.weight"))
            if layer == 1:
                qg = (a @ w("attn_q.weight").T).reshape(4, 2, 16)
                q = rotary(norm(qg[:, 0], w("attn_q_norm.weight")), position)
                k = rotary(
                    norm((a @ w("attn_k.weight").T).reshape(2, 16), w("attn_k_norm.weight")),
                    position,
                )
                v = (a @ w("attn_v.weight").T).reshape(2, 16)
                history.append((k, v))
                keys = np.stack([item[0] for item in history])[:, [0, 0, 1, 1]]
                values = np.stack([item[1] for item in history])[:, [0, 0, 1, 1]]
                scores = np.einsum("hd,thd->ht", q, keys) / 4
                probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                mixed = np.einsum("ht,thd->hd", probabilities, values) * sigmoid(qg[:, 1])
                mixed = mixed.ravel() @ w("attn_output.weight").T
            else:
                previous, delta = recurrent[layer]
                qkv = a @ w("attn_qkv.weight").T
                convolution = np.concatenate((previous, qkv[:, None]), axis=1)
                activated = np.sum(convolution * w("ssm_conv1d.weight"), axis=1)
                activated *= sigmoid(activated)
                q, k, v = np.split(activated.reshape(8, 16), [2, 4])
                q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * 4
                k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                q, k = q[[0, 1, 0, 1]], k[[0, 1, 0, 1]]
                decay = np.exp(
                    w("ssm_a") * np.logaddexp(0, a @ w("ssm_alpha.weight").T + w("ssm_dt.bias"))
                )
                beta = sigmoid(a @ w("ssm_beta.weight").T)
                delta = delta * decay[:, None, None]
                correction = (v - np.einsum("hvk,hk->hv", delta, k)) * beta[:, None]
                delta += correction[:, :, None] * k[:, None, :]
                mixed = norm(np.einsum("hvk,hk->hv", delta, q), w("ssm_norm.weight"))
                gate = a @ w("attn_gate.weight").T
                mixed = (mixed.ravel() * gate * sigmoid(gate)) @ w("ssm_out.weight").T
                recurrent[layer] = convolution[:, 1:], delta
            x = x + mixed
            a = norm(x, w("post_attention_norm.weight"))
            gate, up = a @ w("ffn_gate.weight").T, a @ w("ffn_up.weight").T
            x = x + (gate * sigmoid(gate) * up) @ w("ffn_down.weight").T
        result.append(norm(x, weights["output_norm.weight"]) @ weights["token_embd.weight"].T)
    return np.stack(result), recurrent, history


@pytest.mark.device
def test_hybrid_model_chunk_decode_checkpoint_and_aborted_candidate(tmp_path, monkeypatch):
    path = tmp_path / "hybrid.gguf"
    weights = fixture(path)
    tokens = (1, 3, 2, 7, 8, 12, 14, 3, 6)
    expected, recurrent, _ = reference(weights, tokens)
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    with ExitStack() as cleanup:
        context = open_context(backend, 64 * 1024**2, 0)
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        provider = ResidentOperations(artifact, context)
        cleanup.callback(provider.close)
        description = inspect_dense(artifact.directory, artifact.identity)
        incompatible = description.model_copy(
            update={"artifact_identity": ArtifactIdentity("0" * 64)}
        )
        before_binding = context.allocated_bytes
        with pytest.raises(ValueError, match="different artifacts"):
            DenseRuntime(incompatible, provider)
        assert context.allocated_bytes == before_binding
        model = DenseRuntime(description, provider)
        cleanup.callback(model.close)
        selector = SampleSelector(context)
        cleanup.callback(selector.close)

        def run(state, inputs, commit=True):
            forward = model._prepare_numerical(
                (ForwardRequest(state, Inputs.text(inputs, state.position), LogitsSelection.ALL),)
            )
            try:
                with Preparation(context) as prepared:
                    samples = prepared.allocate(TensorSpec((len(inputs), 2), DType.I32))
                    draws = tuple(
                        Draw(
                            kind=SelectionKind.GREEDY,
                            seed=SamplingSeed(0),
                            position=SamplePosition(state.position + i),
                        )
                        for i in range(len(inputs))
                    )
                    prepared.add(*forward.commands)
                    prepared.add(*selector.prepare(forward.outputs[0].logits, draws, samples))
                    completion = context.submit(prepared.finish())
                    forward.submitted(completion)
                    selected = selector.read(samples, after=completion)
                    actual = np.frombuffer(forward.outputs[0].read_logits(), np.float32).reshape(
                        len(inputs), 32
                    )
                    assert selected == tuple(
                        SampledToken(token=int(token)) for token in actual.argmax(axis=1)
                    )
                    if commit:
                        forward.outputs[0].commit()
                    return actual
            finally:
                forward.close()

        state = model.states.create()
        np.testing.assert_allclose(run(state, tokens), expected, atol=3e-5, rtol=3e-5)
        assert state.position == len(tokens)
        fence = context.submit([])
        for index, (convolution, delta) in recurrent.items():
            numerical = state.recurrent[index]
            actual = np.frombuffer(
                context.read(numerical.convolution, after=fence), np.float32
            ).reshape(128, 3)
            np.testing.assert_allclose(actual, convolution, atol=3e-5, rtol=3e-5)
            actual = np.frombuffer(context.read(numerical.delta, after=fence), np.float32).reshape(
                4, 16, 16
            )
            np.testing.assert_allclose(actual, delta, atol=3e-5, rtol=3e-5)
        state.close()

        state = model.states.create()
        np.testing.assert_allclose(run(state, tokens[:3]), expected[:3], atol=3e-5, rtol=3e-5)
        snapshot = state.checkpoint()
        fork = model.states.create(snapshot)
        # Abandon an already submitted candidate. The accepted state and the
        # checkpoint must still produce the same teacher-forced continuation.
        forward = model._prepare_numerical(
            (ForwardRequest(state, Inputs.text((30, 29), state.position), LogitsSelection.NONE),)
        )
        ticket = context.submit(forward.commands)
        forward.submitted(ticket)
        forward.close()
        ticket.wait()
        assert state.position == fork.position == 3
        builds = []
        original = model.program._commands

        def record_build(*args):
            builds.append(None)
            return original(*args)

        with monkeypatch.context() as patch:
            patch.setattr(model.program, "_commands", record_build)
            for offset in range(3, len(tokens)):
                np.testing.assert_allclose(
                    run(state, tokens[offset : offset + 1]),
                    expected[offset : offset + 1],
                    atol=3e-5,
                    rtol=3e-5,
                )
        assert len(builds) == 1  # Bank changes/visibility updates rebind the same equations.
        alternative = (11, 25, 0)
        branch, _, _ = reference(weights, tokens[:3] + alternative)
        np.testing.assert_allclose(run(fork, alternative), branch[3:], atol=3e-5, rtol=3e-5)
        assert snapshot.position == 3
        model.close()
        provider.close()
        assert context.allocated_bytes == 0


@pytest.mark.device
def test_model_preparation_failure_unwinds_commands_state_and_pool(tmp_path, monkeypatch):
    path = tmp_path / "hybrid.gguf"
    fixture(path)
    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")), 64 * 1024**2, 0
        )
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        provider = ResidentOperations(artifact, context)
        cleanup.callback(provider.close)
        model = DenseRuntime(inspect_dense(artifact.directory, artifact.identity), provider)
        cleanup.callback(model.close)
        state = model.states.create()
        model.program.reserve(3)
        baseline = context.allocated_bytes

        def fail(*_):
            raise RuntimeError("injected late operation preparation failure")

        with monkeypatch.context() as patch:
            patch.setattr(model.program.blocks[1].gate, "prepare", fail)
            with pytest.raises(RuntimeError, match="injected late"):
                model._prepare_numerical(
                    (ForwardRequest(state, Inputs.text((1, 2, 3), state.position)),)
                )
        assert state.position == 0 and state.pending is None
        assert model.states.pool.slab_count == 0
        assert context.allocated_bytes == baseline
        assert not context._pending

        with monkeypatch.context() as patch:
            patch.setattr(context, "budget_bytes", baseline + 128)
            with pytest.raises(CapacityError):
                model._prepare_numerical(
                    (ForwardRequest(state, Inputs.text((1, 2, 3), state.position)),)
                )
        assert state.position == 0 and state.pending is None
        assert context.allocated_bytes == baseline

        forward = model._prepare_numerical(
            (ForwardRequest(state, Inputs.text((1, 2, 3), state.position)),)
        )
        unrelated = context.submit([])
        unrelated.wait()
        with pytest.raises(ValueError, match="every command"):
            forward.submitted(unrelated)
        with pytest.raises(RuntimeError, match="completion"):
            forward.outputs[0].commit()
        forward.close()
        assert state.position == 0 and state.pending is None
        assert model.states.pool.slab_count == 0
        # A successfully captured plan may retain numerical scratch, never
        # request state. Its explicit eviction returns that cache to the budget.
        model.program.release_binding()
        assert context.allocated_bytes == baseline
        model.close()
        provider.close()
        assert context.allocated_bytes == 0


@pytest.mark.device
def test_conditioned_sequence_chunk_checkpoint_fork_and_input_ownership(tmp_path):
    path = tmp_path / "conditioned.gguf"
    weights = fixture(path)
    tokens = (1, 3, 3, 3, 7, 8)
    coordinates = ((0, 0, 0), (1, 1, 1), (1, 1, 2), (1, 2, 1), (3, 3, 3), (4, 4, 4))
    visual = np.random.default_rng(73).normal(0, 0.1, (3, 32)).astype(np.float32)
    continuation = (12, 14)
    embeddings = weights["token_embd.weight"][list(tokens + continuation)].copy()
    embeddings[1:4] = visual
    expected, _, _ = reference(
        weights,
        tokens + continuation,
        embeddings=embeddings,
        coordinates=coordinates + ((5, 5, 5), (6, 6, 6)),
    )
    plan = InputPlan(
        tokens=tokens,
        layout=InputLayout(count=6, spans=(InputSpan(start=1, end=4, identity="content-a"),)),
        coordinates=coordinates,
        continuation=5,
    )
    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")),
            64 * 1024**2,
            0,
        )
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        provider = ResidentOperations(artifact, context)
        cleanup.callback(provider.close)
        model = DenseRuntime(inspect_dense(artifact.directory, artifact.identity), provider)
        cleanup.callback(model.close)
        feature = context.upload(TensorSpec((3, 32), DType.F32), visual.tobytes())
        sequence = model.create(plan, (Feature("content-a", feature),))
        feature.close()  # Model input state owns its own view.

        def run(sequence, inputs, commit=True):
            batch = model.prepare((ModelRequest(sequence, inputs, LogitsSelection.ALL),))
            advance = batch.advances[0]
            try:
                ticket = context.submit(batch.commands)
                batch.submitted(ticket)
                values = np.frombuffer(context.read(advance.logits, after=ticket), np.float32)
                if commit:
                    advance.commit()
                return values.reshape(len(inputs), 32)
            finally:
                batch.close()

        np.testing.assert_allclose(run(sequence, tokens[:2]), expected[:2], atol=3e-5, rtol=3e-5)
        checkpoint = sequence.checkpoint()  # Inside an image: remaining features must survive.
        fork = checkpoint.fork()
        assert len(checkpoint.inputs.features) == len(fork.inputs.features) == 1
        batch = model.prepare(
            tuple(ModelRequest(item, tokens[2:4], LogitsSelection.ALL) for item in (sequence, fork))
        )
        try:
            completion = context.submit(batch.commands)
            batch.submitted(completion)
            for advance in batch.advances:
                actual = np.frombuffer(
                    context.read(advance.logits, after=completion), np.float32
                ).reshape(2, 32)
                np.testing.assert_allclose(actual, expected[2:4], atol=3e-5, rtol=3e-5)
        finally:
            batch.close()  # Both candidates are discarded; each complete input stays at 2.

        assert sequence.position == sequence.inputs.position == 2
        np.testing.assert_allclose(run(sequence, tokens[2:]), expected[2:6], atol=3e-5, rtol=3e-5)
        assert sequence.inputs.features == ()
        sequence.close()
        checkpoint.close()
        np.testing.assert_allclose(run(fork, tokens[2:4]), expected[2:4], atol=3e-5, rtol=3e-5)
        assert fork.inputs.features == ()
        np.testing.assert_allclose(run(fork, tokens[4:]), expected[4:6], atol=3e-5, rtol=3e-5)
        complete = fork.checkpoint()
        fork.close()
        resumed = complete.fork()
        complete.close()
        # Logical KV position is six; rotary continuation is five after the visual span.
        assert resumed.position == 6
        assert resumed.inputs.assemble(continuation).coordinates == ((5, 5, 5), (6, 6, 6))
        np.testing.assert_allclose(run(resumed, continuation), expected[6:], atol=3e-5, rtol=3e-5)
        model.close()
        provider.close()
        assert context.allocated_bytes == 0
