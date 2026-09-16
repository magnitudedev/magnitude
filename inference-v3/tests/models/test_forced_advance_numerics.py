"""A state-only token block must match repeated advances through the portable backend."""

import numpy as np
import pytest

import ops
from engine import DevicePlan
from engine.data import TokenId
from engine.models.qwen35.description import AttentionWeights
from engine.models.qwen35.inputs import InputPlan
from engine.models.qwen35.runtime import DenseRuntime
from engine.models.sequence import LogitsSelection, ModelRequest
from tests.models.test_qwen35_tensor_program import ArrayResidency, _description, _weight


def description():
    original = _description()
    # Packed persistent KV requires complete 32-value blocks. Keep this toy model
    # small while exercising the same typed attention and recurrent state as serving.
    attention = original.blocks[0].model_copy(
        update={
            "mixer": AttentionWeights(
                query_gate=_weight("a.q", (128, 8)),
                key=_weight("a.k", (32, 8)),
                value=_weight("a.v", (32, 8)),
                query_norm=_weight("a.q_norm", (32,)),
                key_norm=_weight("a.k_norm", (32,)),
                output=_weight("a.out", (8, 64)),
            )
        }
    )
    return original.model_copy(
        update={
            "geometry": original.geometry.model_copy(
                update={
                    "attention_width": 32,
                    "rotary_width": 32,
                    "rotary_sections": (8, 8, 0, 0),
                }
            ),
            "blocks": (attention, original.blocks[1]),
        }
    )


@pytest.mark.device
@pytest.mark.parametrize("compact", [False, True])
def test_hybrid_state_only_run_matches_repeated_advances_and_next_logits(monkeypatch, compact):
    if not compact:
        monkeypatch.setattr(
            ops,
            "default_kv_representation",
            lambda key, value: ops.dense_kv(key, value, ops.DType.F16),
        )
    description_ = description()
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend="metal", maximum_bytes=128 << 20)
    ) as device:
        residency = ArrayResidency(device)
        residency.identity = description_.artifact_identity
        model = DenseRuntime(description_, device, residency, max_sequences=3, prefill_rows=128)
        block = model.create(InputPlan.text((TokenId(1),)))
        repeated = None
        with_readout = None

        def advance(sequence, tokens, *, logits=False):
            batch = model.prepare(
                (
                    ModelRequest(
                        sequence,
                        tuple(TokenId(token) for token in tokens),
                        LogitsSelection.LAST if logits else LogitsSelection.NONE,
                        (0, 0, 0, 0, 0, 0) if logits else None,
                    ),
                )
            )
            try:
                batch.completion.wait()
                if logits:
                    data = batch.advances[0].forward.read_logits()
                else:
                    assert batch.logits is None
                    assert batch.advances[0].read_sample() is None
                    data = None
                batch.advances[0].commit()
                states = tuple(
                    (
                        np.frombuffer(
                            device.read(state.convolution, after=batch.completion), np.float16
                        ).copy(),
                        np.frombuffer(
                            device.read(state.delta, after=batch.completion), np.float32
                        ).copy(),
                    )
                    for state in sequence.state.recurrent
                )
                return data, states
            finally:
                batch.close()

        try:
            advance(block, (1,))
            checkpoint = block.checkpoint()
            repeated = checkpoint.fork()
            with_readout = checkpoint.fork()
            checkpoint.close()
            _, block_state = advance(block, (2, 3, 4))
            _, readout_state = advance(with_readout, (2, 3, 4), logits=True)
            for left, right in zip(block_state, readout_state, strict=True):
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(left[1], right[1])
            for token in (2, 3, 4):
                _, repeated_state = advance(repeated, (token,))
            assert block.position == repeated.position == 4
            for left, right in zip(block_state, repeated_state, strict=True):
                # Fresh rows use dense KV; later advances read quantized history.
                # Compact KV therefore has chunk-dependent rounding. Keep a
                # separate explicit bound; same-chunk state-only equality above
                # and dense-KV equivalence remain substantially stronger checks.
                np.testing.assert_allclose(
                    left[0], right[0], rtol=0.01, atol=0.01 if compact else 0.001
                )
                np.testing.assert_allclose(left[1], right[1], rtol=0.02, atol=0.0001)
            block_logits, _ = advance(block, (5,), logits=True)
            repeated_logits, _ = advance(repeated, (5,), logits=True)
            np.testing.assert_allclose(
                np.frombuffer(block_logits, np.float32),
                np.frombuffer(repeated_logits, np.float32),
                rtol=0.02,
                atol=0.002,
            )
        finally:
            if repeated is not None:
                repeated.close()
            if with_readout is not None:
                with_readout.close()
            block.close()
            model.close()
            for resource in residency.resources:
                resource.close()
