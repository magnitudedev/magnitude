"""Independent BF16-storage equations across a matrix-prefill/vector-decode transition.

The oracle uses NumPy equations and explicit round-to-nearest-even BF16 storage.
It neither invokes production numerical helpers nor changes the FP32 oracle.
"""

from contextlib import ExitStack

import numpy as np
import pytest

from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.models.qwen35.artifact import inspect_dense
from magnitude_engine.models.qwen35.inputs import Inputs
from magnitude_engine.models.qwen35.runtime import DenseRuntime, ForwardRequest
from magnitude_engine.models.sequence import LogitsSelection
from magnitude_engine.numerics.policy import NumericalFamily
from magnitude_engine.operations.factory import ResidentOperations
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType
from magnitude_engine.platform.machine import open_context
from tests.numerics.test_qwen35_model import fixture


def bf16(values):
    bits = np.asarray(values, np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


def reference(weights, tokens, prefill, residual_f32=False):
    recurrent = {i: (np.zeros((128, 3)), np.zeros((4, 16, 16))) for i in (0, 2)}
    history, outputs = [], []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def norm(x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * weight

    def rotary(x, position):
        output = x.copy()
        angle = position * (10000.0 ** (-np.arange(4) / 4)).astype(np.float32)
        output[:, :4] = x[:, :4] * np.cos(angle) - x[:, 4:8] * np.sin(angle)
        output[:, 4:8] = x[:, 4:8] * np.cos(angle) + x[:, :4] * np.sin(angle)
        return bf16(output)

    def residual(values):
        return np.asarray(values, np.float32).astype(np.float64) if residual_f32 else bf16(values)

    for position, token in enumerate(tokens):
        matrix = position < prefill

        def linear(x, weight, output=True, matrix=matrix):
            result = x @ (bf16(weight) if matrix else weight).T
            return bf16(result) if output else result

        x = bf16(weights["token_embd.weight"][token])
        for layer in range(3):
            prefix = f"blk.{layer}."

            def w(name, prefix=prefix):
                return weights[prefix + name]

            a = bf16(norm(x, w("attn_norm.weight")))
            if layer == 1:
                qg = linear(a, w("attn_q.weight")).reshape(4, 2, 16)
                q = rotary(norm(qg[:, 0], w("attn_q_norm.weight")), position)
                k = rotary(
                    norm(linear(a, w("attn_k.weight")).reshape(2, 16), w("attn_k_norm.weight")),
                    position,
                )
                v = linear(a, w("attn_v.weight")).reshape(2, 16)
                history.append((k, v))
                keys = np.stack([item[0] for item in history])[:, [0, 0, 1, 1]]
                values = np.stack([item[1] for item in history])[:, [0, 0, 1, 1]]
                scores = np.einsum("hd,thd->ht", q, keys) / 4
                exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
                denominator = exp.sum(axis=-1, keepdims=True)
                if matrix:
                    # This fixture's eight-token prefill fits one probability tile.
                    # Streaming attention rounds the unnormalized exponential
                    # operand, retaining its denominator and accumulation in FP32.
                    mixed = np.einsum("ht,thd->hd", bf16(exp), values) / denominator
                else:
                    mixed = np.einsum("ht,thd->hd", exp, values) / denominator
                mixed = bf16(bf16(mixed) * sigmoid(qg[:, 1]))
                mixed = linear(mixed.ravel(), w("attn_output.weight"))
            else:
                previous, delta = recurrent[layer]
                qkv = linear(a, w("attn_qkv.weight"))
                convolution = np.concatenate((previous, qkv[:, None]), axis=1)
                activated = np.sum(convolution * w("ssm_conv1d.weight"), axis=1)
                activated *= sigmoid(activated)
                q, k, v = np.split(activated.reshape(8, 16), [2, 4])
                q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * 4
                k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                q, k = q[[0, 1, 0, 1]], k[[0, 1, 0, 1]]
                alpha = linear(a, w("ssm_alpha.weight"))
                beta = sigmoid(linear(a, w("ssm_beta.weight")))
                decay = np.exp(w("ssm_a") * np.logaddexp(0, alpha + w("ssm_dt.bias")))
                delta = delta * decay[:, None, None]
                correction = (v - np.einsum("hvk,hk->hv", delta, k)) * beta[:, None]
                delta += correction[:, :, None] * k[:, None, :]
                mixed = bf16(norm(np.einsum("hvk,hk->hv", delta, q), w("ssm_norm.weight")))
                gate = linear(a, w("attn_gate.weight"))
                mixed = linear(bf16(mixed.ravel() * gate * sigmoid(gate)), w("ssm_out.weight"))
                recurrent[layer] = convolution[:, 1:], delta
            x = residual(x + mixed)
            a = bf16(norm(x, w("post_attention_norm.weight")))
            gate, up = linear(a, w("ffn_gate.weight")), linear(a, w("ffn_up.weight"))
            x = residual(x + linear(bf16(gate * sigmoid(gate) * up), w("ffn_down.weight")))
        outputs.append(
            linear(
                residual(norm(x, weights["output_norm.weight"])),
                weights["token_embd.weight"],
                output=False,
                matrix=matrix and not residual_f32,
            )
        )
    return np.stack(outputs), recurrent


@pytest.mark.device
@pytest.mark.parametrize(
    "family", [NumericalFamily.MIXED_BF16, NumericalFamily.MIXED_BF16_F32_RESIDUAL]
)
def test_mixed_storage_prefill_decode_and_checkpoint(tmp_path, family):
    path = tmp_path / "mixed.gguf"
    weights = fixture(path)
    tokens = (1, 3, 2, 7, 8, 12, 14, 3, 6, 11, 9)
    expected, _ = reference(
        weights, tokens, 8, residual_f32=family == NumericalFamily.MIXED_BF16_F32_RESIDUAL
    )
    with ExitStack() as cleanup:
        context = open_context(Backend.METAL, 64 * 1024**2, 0)
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        provider = ResidentOperations(artifact, context)
        cleanup.callback(provider.close)
        model = DenseRuntime(
            inspect_dense(artifact.directory, artifact.identity),
            provider,
            family,
        )
        cleanup.callback(model.close)
        state = model.states.create()
        assert model.states.pool.layout.dtype == DType.BF16
        assert state.recurrent[0].convolution.spec.dtype == DType.BF16
        assert state.recurrent[0].delta.spec.dtype == DType.F32

        def advance(state, inputs):
            forward = model._prepare_numerical(
                (ForwardRequest(state, Inputs.text(inputs, state.position), LogitsSelection.ALL),)
            )
            try:
                ticket = context.submit(forward.commands)
                forward.submitted(ticket)
                values = (
                    np.frombuffer(forward.outputs[0].read_logits(), np.float32)
                    .reshape(-1, 32)
                    .copy()
                )
                forward.outputs[0].commit()
                return values
            finally:
                forward.close()

        actual = [advance(state, tokens[:8])]
        checkpoint = state.checkpoint()
        fork = model.states.create(checkpoint)
        for token in tokens[8:]:
            actual.append(advance(state, (token,)))
        actual = np.concatenate(actual)
        # BF16 storage has unit roundoff 2^-8; allow two such output-scale
        # roundoffs for this three-block fixture. Full-model bounds are separate.
        np.testing.assert_allclose(actual, expected, atol=0.008, rtol=0.008)
        np.testing.assert_array_equal(actual.argmax(axis=-1), expected.argmax(axis=-1))
        replay = np.concatenate([advance(fork, (token,)) for token in tokens[8:]])
        np.testing.assert_array_equal(replay, actual[8:])
        model.close()
        provider.close()
        assert context.allocated_bytes == 0
