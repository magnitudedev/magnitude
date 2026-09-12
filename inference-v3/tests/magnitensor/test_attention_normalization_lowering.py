import numpy as np
import pytest
import torch

import magnitensor as mt
from magnitensor.compiler.lowering import LoweringContext, select_cover

CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    memory_scopes=frozenset({"global", "shared", "local"}),
    native_multi_launch=True,
    partial_binding=True,
    fingerprint="attention-normalization-test",
)


def _selected(graph, mode="prefill"):
    context = LoweringContext(CAPABILITIES, mode, "model", "test", 1 << 20)
    return tuple(
        candidate.name
        for candidate in select_cover(graph, mt.lowerings.enumerate(graph, context)).candidates
    )


def test_residual_and_rms_normalization_form_one_lowering_region():
    value = mt.TensorSpec((3, 16), mt.DType.F16)
    weight = mt.TensorSpec((16,), mt.DType.F16)

    def function(left, right, gain):
        residual = left + right
        return residual, mt.rms_norm(residual, gain, epsilon=1e-6)

    graph = mt.trace(
        function,
        mt.Signature(
            (
                mt.Argument(value, "left"),
                mt.Argument(value, "right"),
                mt.Argument(weight, "gain"),
            )
        ),
    )
    assert _selected(graph) == ("residual_rms.fused@0:1",)


@pytest.mark.device
def test_fused_residual_rms_matches_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal normalization qualification requires MPS")
    rng = np.random.default_rng(32)
    arrays = (
        rng.normal(0, 0.2, (3, 16)).astype(np.float16),
        rng.normal(0, 0.2, (3, 16)).astype(np.float16),
        rng.normal(1, 0.1, (16,)).astype(np.float16),
    )
    specs = tuple(mt.TensorSpec(value.shape, mt.DType.F16) for value in arrays)

    def function(left, right, gain):
        residual = left + right
        return residual, mt.rms_norm(residual, gain, epsilon=1e-6)

    signature = mt.Signature(
        tuple(
            mt.Argument(spec, name)
            for spec, name in zip(specs, ("left", "right", "gain"), strict=True)
        )
    )
    graph = mt.trace(function, signature)
    expected = mt.evaluate_reference(
        graph, dict(zip(("left", "right", "gain"), arrays, strict=True))
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
        execution = compiled.submit(*resources)
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


@pytest.mark.device
def test_parallel_online_attention_and_residual_rms_match_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal lowering qualification requires MPS")
    rng = np.random.default_rng(31)
    query = rng.normal(0, 0.2, (3, 4, 8)).astype(np.float32)
    history = rng.normal(0, 0.2, (2, 1040, 2, 8)).astype(np.float32)
    visible = np.asarray([[0, 3], [2, 37], [7, 1025]], dtype=np.int32)
    specs = tuple(
        mt.TensorSpec(value.shape, dtype)
        for value, dtype in (
            (query, mt.DType.F32),
            (history, mt.DType.F32),
            (visible, mt.DType.I32),
        )
    )
    signature = mt.Signature(
        tuple(
            mt.Argument(
                spec,
                name,
                mt.ValueKind.RESOURCE if name == "kv" else mt.ValueKind.INPUT,
            )
            for spec, name in zip(specs, ("q", "kv", "visible"), strict=True)
        )
    )

    def function(q, kv, reads):
        return mt.causal_attention(q, kv, reads)

    graph = mt.trace(function, signature)
    expected = mt.evaluate_reference(
        graph, {"q": query, "kv": history, "visible": visible}
    ).outputs[0]
    assert _selected(graph) == ("causal_attention.online@0",)

    device = mt.device("metal", budget_bytes=1 << 20)
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes())
            for spec, value in zip(specs, (query, history, visible), strict=True)
        ]
        compiled = mt.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=mt.CompileOptions(mode="prefill"),
        )
        execution = compiled.submit(resources[0], resources[2], resources={"kv": resources[1]})
        execution.completion.wait()
        np.testing.assert_allclose(
            execution.outputs[0].native.cpu().numpy(), expected, rtol=2e-4, atol=2e-4
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


@pytest.mark.device
@pytest.mark.parametrize(
    "mode,rows,capacity,visible,expected_name",
    (
        (
            "prefill",
            8,
            64,
            np.asarray([[0, index + 1] for index in range(8)], dtype=np.int32),
            "causal_attention.matrix-streaming@0",
        ),
        (
            "decode",
            1,
            1024,
            np.asarray([[0, 1000]], dtype=np.int32),
            "causal_attention.partitioned@0",
        ),
    ),
)
def test_optimized_attention_schedules_match_reference_on_metal(
    mode, rows, capacity, visible, expected_name
):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal optimized attention qualification requires MPS")
    rng = np.random.default_rng(45)
    query = rng.normal(0, 0.1, (rows, 4, 8)).astype(np.float16)
    history = rng.normal(0, 0.1, (2, capacity, 2, 8)).astype(np.float16)
    specs = (
        mt.TensorSpec(query.shape, mt.DType.F16),
        mt.TensorSpec(history.shape, mt.DType.F16),
        mt.TensorSpec(visible.shape, mt.DType.I32),
    )
    signature = mt.Signature(
        (
            mt.Argument(specs[0], "query"),
            mt.Argument(specs[1], "history", mt.ValueKind.RESOURCE),
            mt.Argument(specs[2], "visible"),
        )
    )

    def function(q, cache, limits):
        return mt.causal_attention(q, cache, limits, sequence_count=1)

    graph = mt.trace(function, signature)
    expected = mt.evaluate_reference(
        graph,
        {"query": query, "history": history, "visible": visible},
    ).outputs[0]
    device = mt.device("metal", budget_bytes=1 << 28)
    resources = compiled = execution = None
    try:
        resources = (
            device.upload(specs[0], query.tobytes()),
            device.upload(specs[1], history.tobytes()),
            device.upload(specs[2], visible.tobytes()),
        )
        compiled = mt.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=mt.CompileOptions(mode=mode),
        )
        assert compiled.diagnostics.submissions == ((expected_name,),)
        execution = compiled.submit(
            resources[0],
            resources[2],
            resources={"history": resources[1]},
        )
        execution.completion.wait()
        np.testing.assert_allclose(
            execution.outputs[0].native.cpu().numpy(),
            expected,
            rtol=3e-2,
            atol=3e-2,
        )
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        if resources is not None:
            for resource in reversed(resources):
                resource.close()
        device.close()
