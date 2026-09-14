import numpy as np
import pytest
import torch

import magnitensor as mt
from magnitensor.compiler.lowering import LoweringContext, select_cover

CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    matrix_instructions=(
        mt.MatrixInstruction(8, 8, 8, mt.DType.F16, mt.DType.F32),
        mt.MatrixInstruction(8, 8, 8, mt.DType.F32, mt.DType.F32),
    ),
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


def test_streaming_attention_workspace_is_bounded_by_partition_outputs():
    rows, heads, width, capacity = 129, 4, 256, 8192
    query = mt.TensorSpec((rows, heads, width), mt.DType.F16)
    history = mt.TensorSpec((2, capacity, 1, width), mt.DType.F16)
    visible = mt.TensorSpec((rows, 2), mt.DType.I32)
    weight = mt.TensorSpec((32, heads * width), mt.DType.F16).with_representation(
        mt.Affine(mt.Code(4), 64, mt.DirectCoefficients(mt.DType.BF16, mt.DType.BF16))
    )

    def function(q, kv, reads, gate, projection):
        attended = mt.causal_attention(q, kv, reads, sequence_count=1)
        return mt.linear(mt.reshape(attended * mt.sigmoid(gate), (rows, heads * width)), projection)

    graph = mt.trace(
        function,
        mt.Signature(
            tuple(
                mt.Argument(
                    spec, name, mt.ValueKind.RESOURCE if name == "kv" else mt.ValueKind.INPUT
                )
                for spec, name in zip(
                    (query, history, visible, query, weight),
                    ("q", "kv", "reads", "gate", "weight"),
                    strict=True,
                )
            )
        ),
    )
    for budget in (16 << 20, 4 << 20):
        context = LoweringContext(CAPABILITIES, "prefill", "model", "test", budget)
        cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
        assert len(cover.candidates) == 1
        candidate = cover.candidates[0]
        assert candidate.name.startswith("attention.matrix-streaming-gated-output")
        assert candidate.workspace_bytes <= budget
        assert candidate.kernel_count == 3


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
def test_bfloat_residual_rms_uses_the_published_residual_for_both_moments():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal normalization qualification requires MPS")
    # The exact sum lies below the midpoint between BF16 values. Keeping that
    # sum only in the denominator used to change normalized output by one ULP.
    left = torch.ones((2, 2048), dtype=torch.bfloat16)
    right = torch.full_like(left, 0.003)
    gain = torch.ones((2048,), dtype=torch.bfloat16)
    arrays = (left, right, gain)
    specs = tuple(mt.TensorSpec(tuple(value.shape), mt.DType.BF16) for value in arrays)
    signature = mt.Signature(
        tuple(
            mt.Argument(spec, name)
            for spec, name in zip(specs, ("left", "right", "gain"), strict=True)
        )
    )

    def function(a, b, weight):
        residual = a + b
        return residual, mt.rms_norm(residual, weight, epsilon=1e-6)

    device = mt.device("metal", budget_bytes=1 << 20)
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.view(torch.uint16).numpy().tobytes())
            for spec, value in zip(specs, arrays, strict=True)
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
        expected = left + right
        torch.testing.assert_close(execution.outputs[0].native.cpu(), expected, rtol=0, atol=0)
        normalized = (
            expected.float() * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        torch.testing.assert_close(execution.outputs[1].native.cpu(), normalized, rtol=0, atol=0)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


def test_decode_attention_without_qualified_register_geometry_is_uncovered():
    query = mt.TensorSpec((1, 4, 8), mt.DType.F16)
    history = mt.TensorSpec((2, 1040, 2, 8), mt.DType.F16)
    visible = mt.TensorSpec((1, 2), mt.DType.I32)
    graph = mt.trace(
        lambda q, kv, reads: mt.causal_attention(q, kv, reads),
        mt.Signature(
            (
                mt.Argument(query, "q"),
                mt.Argument(history, "kv", mt.ValueKind.RESOURCE),
                mt.Argument(visible, "visible"),
            )
        ),
    )

    with pytest.raises(ValueError, match="no legal lowering"):
        _selected(graph, mode="decode")


@pytest.mark.device
@pytest.mark.parametrize(
    "mode,rows,capacity,width,visible,expected_name",
    (
        (
            "prefill",
            8,
            64,
            8,
            np.asarray([[0, index + 1] for index in range(8)], dtype=np.int32),
            "causal_attention.matrix-streaming@0",
        ),
        (
            "decode",
            1,
            1024,
            256,
            np.asarray([[0, 1000]], dtype=np.int32),
            "causal_attention.register-partitioned@0",
        ),
        (
            "prefill",
            19,
            8192,
            256,
            np.asarray([[0, 8000 + index] for index in range(19)], dtype=np.int32),
            "causal_attention.matrix-streaming@0",
        ),
    ),
)
def test_optimized_attention_schedules_match_reference_on_metal(
    mode, rows, capacity, width, visible, expected_name
):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal optimized attention qualification requires MPS")
    rng = np.random.default_rng(45)
    query = rng.normal(0, 0.1, (rows, 4, width)).astype(np.float16)
    # Four query heads share one KV head. The long-history case crosses a
    # query-tile boundary and ends in both a query and a value-subtile tail.
    history = rng.normal(0, 0.1, (2, capacity, 1, width)).astype(np.float16)
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
