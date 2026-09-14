import numpy as np
import pytest
import torch

import ops
from ops.operation import build_operations
from engine import DevicePlan
from ops.compiler.lowering import LoweringContext, plan_submissions
from ops.compiler.memory import plan_memory
from ops.compiler.unit import build_unit
from ops.runtime.tilelang import _build_reusable_module

CAPABILITIES = ops.Capabilities(
    32,
    256,
    32 * 1024,
    memory_scopes=frozenset({"global", "shared", "local"}),
    native_multi_launch=True,
    partial_binding=True,
)


def _construct(graph, mode="decode"):
    context = LoweringContext(CAPABILITIES, mode, "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    memory = plan_memory(graph, cover, CAPABILITIES)
    submission = plan_submissions(graph, cover, CAPABILITIES)
    assert len(submission) == 1
    return _build_reusable_module(build_unit(graph, memory, submission[0]))


def test_attention_prepare_has_reference_and_standalone_physical_body():
    specs = (
        ops.TensorSpec((2, 16), ops.DType.F32),
        ops.TensorSpec((2, 4), ops.DType.F32),
        ops.TensorSpec((4,), ops.DType.F32),
        ops.TensorSpec((4,), ops.DType.F32),
        ops.TensorSpec((2, 3), ops.DType.I32),
    )
    signature = ops.Signature(tuple(ops.Argument(spec, f"v{i}") for i, spec in enumerate(specs)))

    def function(query_gate, keys, query_norm, key_norm, coordinates):
        return ops.attention_prepare(
            query_gate,
            keys,
            query_norm,
            key_norm,
            coordinates,
            query_heads=2,
            kv_heads=1,
            width=4,
            rotary_width=4,
            base=10_000.0,
            sections=(1, 1, 0, 0),
            epsilon=1e-6,
        )

    graph = ops.trace(function, signature)
    rng = np.random.default_rng(3)
    values = {
        "v0": rng.normal(size=specs[0].shape).astype(np.float32),
        "v1": rng.normal(size=specs[1].shape).astype(np.float32),
        "v2": np.ones(specs[2].shape, np.float32),
        "v3": np.ones(specs[3].shape, np.float32),
        "v4": np.asarray([[0, 0, 0], [1, 2, 3]], np.int32),
    }
    outputs = ops.evaluate_reference(graph, values).outputs
    assert tuple(output.shape for output in outputs) == ((2, 2, 4), (2, 1, 4), (2, 2, 4))
    # Independent four-channel RoPE oracle: the two frequencies are 1 and
    # 1/sqrt(10000), with coordinates drawn from the first and second axes.
    # Shape-only coverage missed an exponent denominator of half the width.
    angles = np.asarray([[0.0, 0.0], [1.0, 0.02]], np.float32)[:, None, :]
    raw_query = values["v0"].reshape(2, 2, 2, 4)
    for raw, actual in zip(
        (raw_query[:, :, 0], values["v1"].reshape(2, 1, 4)), outputs[:2], strict=True
    ):
        normalized = raw / np.sqrt(np.mean(raw * raw, axis=-1, keepdims=True) + 1e-6)
        first, second = normalized[..., :2], normalized[..., 2:]
        expected = np.concatenate(
            (
                first * np.cos(angles) - second * np.sin(angles),
                second * np.cos(angles) + first * np.sin(angles),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(outputs[2], raw_query[:, :, 1])
    assert _construct(graph).functions


def test_gated_delta_recurrence_preserves_state_as_an_explicit_output():
    specs = (
        ops.TensorSpec((2, 1, 4), ops.DType.F32),
        ops.TensorSpec((2, 1, 4), ops.DType.F32),
        ops.TensorSpec((2, 2, 3), ops.DType.F32),
        ops.TensorSpec((2, 2), ops.DType.F32),
        ops.TensorSpec((2, 2), ops.DType.F32),
        ops.TensorSpec((1, 2, 3, 4), ops.DType.F32),
        ops.TensorSpec((2,), ops.DType.I32),
    )
    signature = ops.Signature(
        tuple(
            ops.Argument(spec, f"v{i}", ops.ValueKind.RESOURCE if i == 5 else ops.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )
    graph = ops.trace(
        lambda q, k, v, decay, beta, state, offsets: ops.gated_delta_recurrence(
            q, k, v, decay, beta, state, offsets, mapping="tiled"
        ),
        signature,
    )
    values = {
        f"v{i}": (np.asarray([0, 2], np.int32) if i == 6 else np.ones(spec.shape, np.float32) * 0.1)
        for i, spec in enumerate(specs)
    }
    output, state = ops.evaluate_reference(graph, values).outputs
    assert output.shape == specs[2].shape
    assert state.shape == specs[5].shape
    assert not np.shares_memory(state, values["v5"])
    assert _construct(graph).functions


@pytest.mark.device
def test_gated_delta_recurrence_matches_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal recurrent transition check requires MPS")
    specs = (
        ops.TensorSpec((3, 1, 4), ops.DType.F32),
        ops.TensorSpec((3, 1, 4), ops.DType.F32),
        ops.TensorSpec((3, 2, 4), ops.DType.F32),
        ops.TensorSpec((3, 2), ops.DType.F32),
        ops.TensorSpec((3, 2), ops.DType.F32),
        ops.TensorSpec((2, 2, 4, 4), ops.DType.F32),
        ops.TensorSpec((3,), ops.DType.I32),
    )
    signature = ops.Signature(
        tuple(
            ops.Argument(spec, f"v{i}", ops.ValueKind.RESOURCE if i == 5 else ops.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )

    def function(q, k, v, decay, beta, state, offsets):
        return ops.gated_delta_recurrence(q, k, v, decay, beta, state, offsets, mapping="tiled")

    graph = ops.trace(function, signature)
    rng = np.random.default_rng(28)
    arrays = tuple(
        np.asarray([0, 1, 3], np.int32)
        if index == 6
        else rng.normal(0, 0.1, spec.shape).astype(np.float32)
        for index, spec in enumerate(specs)
    )
    expected = ops.evaluate_reference(
        graph, {f"v{i}": value for i, value in enumerate(arrays)}
    ).outputs
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes()) for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode="prefill"),
        )
        execution = compiled.submit(*resources[:5], resources[6], resources={"v5": resources[5]})
        execution.completion.wait()
        for actual, reference in zip(execution.outputs, expected, strict=True):
            np.testing.assert_allclose(actual.native.cpu().numpy(), reference, rtol=3e-3, atol=3e-3)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


def test_recurrent_prepare_constructs_all_state_and_activation_outputs():
    specs = (
        ops.TensorSpec((2, 12), ops.DType.F32),
        ops.TensorSpec((12, 3), ops.DType.F32),
        ops.TensorSpec((1, 12, 2), ops.DType.F32),
        ops.TensorSpec((2, 1), ops.DType.F32),
        ops.TensorSpec((2, 1), ops.DType.F32),
        ops.TensorSpec((1,), ops.DType.F32),
        ops.TensorSpec((1,), ops.DType.F32),
        ops.TensorSpec((2,), ops.DType.I32),
    )
    signature = ops.Signature(
        tuple(
            ops.Argument(spec, f"v{i}", ops.ValueKind.RESOURCE if i == 2 else ops.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )
    graph = ops.trace(
        lambda x, conv, state, alpha, beta, rate, bias, offsets: ops.recurrent_prepare(
            x,
            conv,
            state,
            alpha,
            beta,
            rate,
            bias,
            offsets,
            key_heads=1,
            value_heads=1,
            width=4,
            convolution_width=3,
            epsilon=1e-6,
        ),
        signature,
    )
    rng = np.random.default_rng(4)
    values = {
        f"v{i}": (
            np.asarray([0, 2], np.int32)
            if i == 7
            else rng.normal(0, 0.1, spec.shape).astype(np.float32)
        )
        for i, spec in enumerate(specs)
    }
    outputs = ops.evaluate_reference(graph, values).outputs
    assert tuple(output.shape for output in outputs) == (
        (2, 1, 4),
        (2, 1, 4),
        (2, 1, 4),
        (2, 1),
        (2, 1),
        (1, 12, 2),
    )
    assert _construct(graph, "prefill").functions


@pytest.mark.device
def test_channel_parallel_recurrent_prepare_matches_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal recurrent preparation check requires MPS")
    specs = (
        ops.TensorSpec((4, 24), ops.DType.F32),
        ops.TensorSpec((24, 3), ops.DType.F32),
        ops.TensorSpec((2, 24, 2), ops.DType.F32),
        ops.TensorSpec((4, 2), ops.DType.F32),
        ops.TensorSpec((4, 2), ops.DType.F32),
        ops.TensorSpec((2,), ops.DType.F32),
        ops.TensorSpec((2,), ops.DType.F32),
        ops.TensorSpec((3,), ops.DType.I32),
    )
    signature = ops.Signature(
        tuple(
            ops.Argument(spec, f"v{i}", ops.ValueKind.RESOURCE if i == 2 else ops.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )

    def function(x, conv, state, alpha, beta, rate, bias, offsets):
        return ops.recurrent_prepare(
            x,
            conv,
            state,
            alpha,
            beta,
            rate,
            bias,
            offsets,
            key_heads=1,
            value_heads=2,
            width=6,
            convolution_width=3,
            epsilon=1e-6,
        )

    graph = ops.trace(function, signature)
    rng = np.random.default_rng(29)
    arrays = tuple(
        np.asarray([0, 1, 4], np.int32)
        if index == 7
        else rng.normal(0, 0.1, spec.shape).astype(np.float32)
        for index, spec in enumerate(specs)
    )
    expected = ops.evaluate_reference(
        graph, {f"v{i}": value for i, value in enumerate(arrays)}
    ).outputs
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes()) for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode="prefill"),
        )
        assert compiled.diagnostics.submissions == (("recurrent_prepare.channel-parallel@0",),)
        execution = compiled.submit(
            resources[0],
            resources[1],
            *resources[3:],
            resources={"v2": resources[2]},
        )
        execution.completion.wait()
        for actual, reference in zip(execution.outputs, expected, strict=True):
            np.testing.assert_allclose(actual.native.cpu().numpy(), reference, rtol=3e-3, atol=3e-3)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


def test_sampling_reference_defines_greedy_categorical_and_failure_semantics():
    logits_spec = ops.TensorSpec((4, 4), ops.DType.F32)
    draws_spec = ops.TensorSpec((4, 6), ops.DType.U32)
    graph = ops.trace(
        lambda logits, draws: ops.sample(logits, draws),
        ops.Signature((ops.Argument(logits_spec, "logits"), ops.Argument(draws_spec, "draws"))),
    )
    logits = np.asarray(
        [
            [1.0, 3.0, 3.0, -np.inf],
            [0.0, 0.0, 0.0, 0.0],
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [0.0, np.nan, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    draws = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 17, 0, 9, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    selected = ops.evaluate_reference(graph, {"logits": logits, "draws": draws}).outputs[0]
    assert selected[0].tolist() == [1, 0]
    assert selected[1, 1] == 0
    assert selected[2].tolist() == [-1, 1]
    assert selected[3].tolist() == [-1, 2]
    assert np.array_equal(
        selected,
        ops.evaluate_reference(graph, {"logits": logits, "draws": draws}).outputs[0],
    )
