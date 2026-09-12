import numpy as np
import pytest
import torch

import magnitensor as mt
from magnitensor.compiler.lowering import LoweringContext, plan_submissions, select_cover
from magnitensor.compiler.memory import plan_memory
from magnitensor.compiler.unit import build_unit
from magnitensor.runtime.tilelang import _build_prim_func

CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    memory_scopes=frozenset({"global", "shared", "local"}),
    native_multi_launch=True,
    partial_binding=True,
)


def _construct(graph, mode="decode"):
    context = LoweringContext(CAPABILITIES, mode, "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    memory = plan_memory(graph, cover, CAPABILITIES)
    submission = plan_submissions(graph, cover, CAPABILITIES)
    assert len(submission) == 1
    return _build_prim_func(build_unit(graph, memory, submission[0]))


def test_attention_prepare_has_reference_and_portable_construction():
    specs = (
        mt.TensorSpec((2, 16), mt.DType.F32),
        mt.TensorSpec((2, 4), mt.DType.F32),
        mt.TensorSpec((4,), mt.DType.F32),
        mt.TensorSpec((4,), mt.DType.F32),
        mt.TensorSpec((2, 3), mt.DType.I32),
    )
    signature = mt.Signature(tuple(mt.Argument(spec, f"v{i}") for i, spec in enumerate(specs)))

    def function(query_gate, keys, query_norm, key_norm, coordinates):
        return mt.attention_prepare(
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

    graph = mt.trace(function, signature)
    rng = np.random.default_rng(3)
    values = {
        "v0": rng.normal(size=specs[0].shape).astype(np.float32),
        "v1": rng.normal(size=specs[1].shape).astype(np.float32),
        "v2": np.ones(specs[2].shape, np.float32),
        "v3": np.ones(specs[3].shape, np.float32),
        "v4": np.asarray([[0, 0, 0], [1, 2, 3]], np.int32),
    }
    outputs = mt.evaluate_reference(graph, values).outputs
    assert tuple(output.shape for output in outputs) == ((2, 2, 4), (2, 1, 4), (2, 2, 4))
    assert _construct(graph).attrs["global_symbol"]


def test_gated_delta_recurrence_preserves_state_as_an_explicit_output():
    specs = (
        mt.TensorSpec((2, 1, 4), mt.DType.F32),
        mt.TensorSpec((2, 1, 4), mt.DType.F32),
        mt.TensorSpec((2, 2, 3), mt.DType.F32),
        mt.TensorSpec((2, 2), mt.DType.F32),
        mt.TensorSpec((2, 2), mt.DType.F32),
        mt.TensorSpec((1, 2, 3, 4), mt.DType.F32),
        mt.TensorSpec((2,), mt.DType.I32),
    )
    signature = mt.Signature(
        tuple(
            mt.Argument(spec, f"v{i}", mt.ValueKind.RESOURCE if i == 5 else mt.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )
    graph = mt.trace(
        lambda q, k, v, decay, beta, state, offsets: mt.gated_delta_recurrence(
            q, k, v, decay, beta, state, offsets, mapping="tiled"
        ),
        signature,
    )
    values = {
        f"v{i}": (
            np.asarray([0, 2], np.int32)
            if i == 6
            else np.ones(spec.shape, np.float32) * 0.1
        )
        for i, spec in enumerate(specs)
    }
    output, state = mt.evaluate_reference(graph, values).outputs
    assert output.shape == specs[2].shape
    assert state.shape == specs[5].shape
    assert not np.shares_memory(state, values["v5"])
    assert _construct(graph).attrs["global_symbol"]


@pytest.mark.device
def test_gated_delta_recurrence_matches_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal recurrent transition check requires MPS")
    specs = (
        mt.TensorSpec((2, 1, 4), mt.DType.F32),
        mt.TensorSpec((2, 1, 4), mt.DType.F32),
        mt.TensorSpec((2, 2, 4), mt.DType.F32),
        mt.TensorSpec((2, 2), mt.DType.F32),
        mt.TensorSpec((2, 2), mt.DType.F32),
        mt.TensorSpec((1, 2, 4, 4), mt.DType.F32),
        mt.TensorSpec((2,), mt.DType.I32),
    )
    signature = mt.Signature(
        tuple(
            mt.Argument(spec, f"v{i}", mt.ValueKind.RESOURCE if i == 5 else mt.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )

    def function(q, k, v, decay, beta, state, offsets):
        return mt.gated_delta_recurrence(q, k, v, decay, beta, state, offsets, mapping="tiled")

    graph = mt.trace(function, signature)
    rng = np.random.default_rng(28)
    arrays = tuple(
        np.asarray([0, 2], np.int32)
        if index == 6
        else rng.normal(0, 0.1, spec.shape).astype(np.float32)
        for index, spec in enumerate(specs)
    )
    expected = mt.evaluate_reference(
        graph, {f"v{i}": value for i, value in enumerate(arrays)}
    ).outputs
    device = mt.device("metal", budget_bytes=1 << 20)
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes())
            for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = mt.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=mt.CompileOptions(mode="prefill"),
        )
        execution = compiled.submit(*resources[:5], resources[6], resources={"v5": resources[5]})
        execution.completion.wait()
        for actual, reference in zip(execution.outputs, expected, strict=True):
            np.testing.assert_allclose(
                actual.native.cpu().numpy(), reference, rtol=3e-3, atol=3e-3
            )
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
        mt.TensorSpec((2, 12), mt.DType.F32),
        mt.TensorSpec((12, 3), mt.DType.F32),
        mt.TensorSpec((1, 12, 2), mt.DType.F32),
        mt.TensorSpec((2, 1), mt.DType.F32),
        mt.TensorSpec((2, 1), mt.DType.F32),
        mt.TensorSpec((1,), mt.DType.F32),
        mt.TensorSpec((1,), mt.DType.F32),
        mt.TensorSpec((2,), mt.DType.I32),
    )
    signature = mt.Signature(
        tuple(
            mt.Argument(spec, f"v{i}", mt.ValueKind.RESOURCE if i == 2 else mt.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )
    graph = mt.trace(
        lambda x, conv, state, alpha, beta, rate, bias, offsets: mt.recurrent_prepare(
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
    outputs = mt.evaluate_reference(graph, values).outputs
    assert tuple(output.shape for output in outputs) == (
        (2, 1, 4),
        (2, 1, 4),
        (2, 1, 4),
        (2, 1),
        (2, 1),
        (1, 12, 2),
    )
    assert _construct(graph, "prefill").attrs["global_symbol"]


@pytest.mark.device
def test_channel_parallel_recurrent_prepare_matches_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal recurrent preparation check requires MPS")
    specs = (
        mt.TensorSpec((4, 24), mt.DType.F32),
        mt.TensorSpec((24, 3), mt.DType.F32),
        mt.TensorSpec((2, 24, 2), mt.DType.F32),
        mt.TensorSpec((4, 2), mt.DType.F32),
        mt.TensorSpec((4, 2), mt.DType.F32),
        mt.TensorSpec((2,), mt.DType.F32),
        mt.TensorSpec((2,), mt.DType.F32),
        mt.TensorSpec((3,), mt.DType.I32),
    )
    signature = mt.Signature(
        tuple(
            mt.Argument(spec, f"v{i}", mt.ValueKind.RESOURCE if i == 2 else mt.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )

    def function(x, conv, state, alpha, beta, rate, bias, offsets):
        return mt.recurrent_prepare(
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

    graph = mt.trace(function, signature)
    rng = np.random.default_rng(29)
    arrays = tuple(
        np.asarray([0, 1, 4], np.int32)
        if index == 7
        else rng.normal(0, 0.1, spec.shape).astype(np.float32)
        for index, spec in enumerate(specs)
    )
    expected = mt.evaluate_reference(
        graph, {f"v{i}": value for i, value in enumerate(arrays)}
    ).outputs
    device = mt.device("metal", budget_bytes=1 << 20)
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes()) for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = mt.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=mt.CompileOptions(mode="prefill"),
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
    logits_spec = mt.TensorSpec((4, 4), mt.DType.F32)
    draws_spec = mt.TensorSpec((4, 6), mt.DType.U32)
    graph = mt.trace(
        lambda logits, draws: mt.sample(logits, draws),
        mt.Signature((mt.Argument(logits_spec, "logits"), mt.Argument(draws_spec, "draws"))),
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
    selected = mt.evaluate_reference(graph, {"logits": logits, "draws": draws}).outputs[0]
    assert selected[0].tolist() == [1, 0]
    assert selected[1, 1] == 0
    assert selected[2].tolist() == [-1, 1]
    assert selected[3].tolist() == [-1, 2]
    assert np.array_equal(
        selected,
        mt.evaluate_reference(graph, {"logits": logits, "draws": draws}).outputs[0],
    )
