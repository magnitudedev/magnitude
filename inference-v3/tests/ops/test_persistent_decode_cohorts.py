"""Attention cohorts cover logical KV groups and every output channel."""

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.kv_codecs import decode_kv_reference, encode_kv_reference
from tests.ops.definitions import implement


@pytest.mark.device
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize(
    "group,kv_heads,width,gated",
    [(group, 2, 64, False) for group in (1, 2, 3, 4, 8, 12, 16)]
    + [
        (4, 1, 256, False),
        (8, 1, 256, False),
        (4, 2, 512, False),
        (4, 2, 512, True),
        (4, 2, 16, False),
        (4, 2, 96, False),
        (4, 2, 192, False),
    ],
)
def test_persistent_attention_covers_group_and_channel_geometry(
    group, kv_heads, width, gated, mode
):
    tokens = 3 if mode == "prefill" else 1
    heads = group * kv_heads
    backend = "cuda" if torch.cuda.is_available() else "metal"
    if backend == "metal" and not torch.backends.mps.is_available():
        pytest.skip("requires a native device")
    rng = np.random.default_rng(81)
    rep = ops.default_kv_representation(width, width)
    history_spec = ops.kv_state_spec(17, kv_heads, ops.DType.F16, rep)
    physical = encode_kv_reference(rng.normal(0, 0.3, history_spec.shape).astype(np.float32), rep)
    history = decode_kv_reference(physical, history_spec)
    query = rng.normal(0, 0.3, (tokens, heads, width)).astype(np.float16)
    keys, values = [
        rng.normal(0, 0.3, (tokens, kv_heads, width)).astype(np.float16) for _ in range(2)
    ]
    visible = np.array([[1, 15, 0, i + 1] for i in range(tokens)], np.int32)
    arrays = (query, keys, values, visible)
    specs = tuple(
        ops.TensorSpec(a.shape, ops.DType.I32 if a.dtype == np.int32 else ops.DType.F16)
        for a in arrays
    )
    signature = ops.Signature(
        tuple(ops.Argument(s, n) for s, n in zip(specs, ("q", "k", "v", "visible"), strict=True))
        + (ops.Argument(history_spec, "history", ops.ValueKind.RESOURCE),)
    )

    def function(q, k, v, visible, history):
        attended = ops.persistent_attention(q, history, k, v, visible, sequence_count=1)
        return (
            ops.reshape(attended * ops.sigmoid(q), (tokens, heads * width)) if gated else attended
        )

    function = implement(function, ops.operation_bodies.attention_mixer)
    graph = ops.trace(function, signature)
    expected = ops.evaluate_reference(
        graph, dict(q=query, k=keys, v=values, visible=visible, history=history)
    ).outputs[0]
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend=backend, maximum_bytes=1 << 29))
    resources, compiled, execution = [], None, None
    try:
        resources = [
            device.upload(spec, array.tobytes()) for spec, array in zip(specs, arrays, strict=True)
        ]
        cache = device.upload(history_spec, physical)
        resources.append(cache)
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode=mode),
        )
        execution = compiled.submit(*resources[:-1], resources={"history": cache})
        execution.completion.wait()
        actual = np.frombuffer(device.read(execution.outputs[0]), dtype=np.float16).reshape(
            expected.shape
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-3)
    finally:
        if execution:
            for output in execution.outputs:
                output.close()
        if compiled:
            compiled.close()
        for resource in resources:
            resource.close()
        device.close()
