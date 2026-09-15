import numpy as np
from dataclasses import replace
import pytest
import torch

import ops
from tests.ops.definitions import implement
from ops.operation import build_operations
from engine import DevicePlan
from ops.compiler.lowering import LoweringContext

CAPABILITIES = ops.Capabilities(
    32,
    256,
    32 * 1024,
    matrix_instructions=(
        ops.MatrixInstruction(8, 8, 8, ops.DType.F16, ops.DType.F32),
        ops.MatrixInstruction(8, 8, 8, ops.DType.F32, ops.DType.F32),
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
        for candidate in build_operations(graph, context)
    )


def test_residual_and_rms_normalization_form_one_lowering_region():
    value = ops.TensorSpec((3, 16), ops.DType.F16)
    weight = ops.TensorSpec((16,), ops.DType.F16)

    def function(left, right, gain):
        residual = left + right
        return residual, ops.rms_norm(residual, gain, epsilon=1e-6)

    function = implement(function, ops.operation_bodies.residual_normalization)

    graph = ops.trace(
        function,
        ops.Signature(
            (
                ops.Argument(value, "left"),
                ops.Argument(value, "right"),
                ops.Argument(weight, "gain"),
            )
        ),
    )
    assert _selected(graph) == ("residual_rms.fused@0:1",)


@pytest.mark.parametrize("capacity", [4096, 8192, 8319])
@pytest.mark.parametrize("threads", [256, 1024])
def test_streaming_attention_workspace_is_bounded_by_partition_outputs(capacity, threads):
    from ops.kernels.attention import CausalAttentionRule

    rows, heads, width = 129, 4, 256
    query = ops.TensorSpec((rows, heads, width), ops.DType.F16)
    history = ops.TensorSpec((2, capacity, 1, width), ops.DType.F16)
    visible = ops.TensorSpec((rows, 2), ops.DType.I32)
    weight = ops.TensorSpec((32, heads * width), ops.DType.F16).with_representation(
        ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16))
    )

    def function(q, kv, reads, gate, projection):
        attended = ops.causal_attention(q, kv, reads, sequence_count=1)
        return ops.linear(ops.reshape(attended * ops.sigmoid(gate), (rows, heads * width)), projection)

    function = implement(function, ops.operation_bodies.attention_mixer)

    graph = ops.trace(
        function,
        ops.Signature(
            tuple(
                ops.Argument(
                    spec, name, ops.ValueKind.RESOURCE if name == "kv" else ops.ValueKind.INPUT
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
        context = LoweringContext(replace(CAPABILITIES, threads_per_group=threads),
                                  "prefill", "model", "test", budget)
        cover = build_operations(graph, context)
        assert len(cover) == 1
        candidate = cover[0]
        assert candidate.name.startswith("attention.matrix-streaming-gated-output")
        assert candidate.workspace_bytes <= budget
        assert candidate.kernel_count == (2 if capacity <= 4096 else 3)
        isolated, = CausalAttentionRule().build(graph, 0, context)
        schedule = candidate.emitter.schedule
        assert schedule == isolated.emitter.schedule
        assert schedule.tile == (32, 32, 256)
        assert schedule.head_tile == 2
        assert schedule.value_tile == 64
        assert schedule.instruction_k == 8
        assert schedule.shared_bytes == 21_120


@pytest.mark.parametrize(
    "heads,threads,dtype,shared,expected",
    [
        (8, 256, ops.DType.F16, 21_120, (2, 32, 21_120)),
        (6, 256, ops.DType.F16, 32_768, (1, 32, 21_120)),
        (8, 128, ops.DType.F16, 32_768, (1, 32, 21_120)),
        (8, 256, ops.DType.F32, 32_768, (2, 16, 25_344)),
        (8, 256, ops.DType.F16, 8_191, None),
    ],
)
def test_attention_staging_respects_head_groups_and_physical_capacity(heads, threads, dtype, shared, expected):
    from ops.kernels.attention import _matrix_attention_schedule

    context = LoweringContext(
        replace(CAPABILITIES, threads_per_group=threads, shared_memory_bytes=shared),
        "prefill", "model", "test", 16 << 20,
    )
    schedule = _matrix_attention_schedule(
        ops.TensorSpec((35, heads, 256), dtype),
        ops.TensorSpec((2, 8192, 2, 256), dtype), context, 1,
    )
    if expected is None:
        assert schedule is None
    else:
        assert schedule is not None
        assert (schedule.head_tile, schedule.tile[1], schedule.shared_bytes) == expected


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
    specs = tuple(ops.TensorSpec(value.shape, ops.DType.F16) for value in arrays)

    def function(left, right, gain):
        residual = left + right
        return residual, ops.rms_norm(residual, gain, epsilon=1e-6)

    function = implement(function, ops.operation_bodies.residual_normalization)

    signature = ops.Signature(
        tuple(
            ops.Argument(spec, name)
            for spec, name in zip(specs, ("left", "right", "gain"), strict=True)
        )
    )
    graph = ops.trace(function, signature)
    expected = ops.evaluate_reference(
        graph, dict(zip(("left", "right", "gain"), arrays, strict=True))
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
    specs = tuple(ops.TensorSpec(tuple(value.shape), ops.DType.BF16) for value in arrays)
    signature = ops.Signature(
        tuple(
            ops.Argument(spec, name)
            for spec, name in zip(specs, ("left", "right", "gain"), strict=True)
        )
    )

    def function(a, b, weight):
        residual = a + b
        return residual, ops.rms_norm(residual, weight, epsilon=1e-6)

    function = implement(function, ops.operation_bodies.residual_normalization)

    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.view(torch.uint16).numpy().tobytes())
            for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode="prefill"),
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
    query = ops.TensorSpec((1, 4, 8), ops.DType.F16)
    history = ops.TensorSpec((2, 1040, 2, 8), ops.DType.F16)
    visible = ops.TensorSpec((1, 2), ops.DType.I32)
    graph = ops.trace(
        lambda q, kv, reads: ops.causal_attention(q, kv, reads),
        ops.Signature(
            (
                ops.Argument(query, "q"),
                ops.Argument(history, "kv", ops.ValueKind.RESOURCE),
                ops.Argument(visible, "visible"),
            )
        ),
    )

    with pytest.raises(ValueError, match="legal tiled matrix or subgroup reduction geometry"):
        _selected(graph, mode="decode")


@pytest.mark.device
@pytest.mark.parametrize(
    "mode,rows,capacity,width,visible,expected_name,heads,kv_heads,floating",
    (
        (
            "prefill",
            8,
            64,
            8,
            np.asarray([[0, index + 1] for index in range(8)], dtype=np.int32),
            "causal_attention.matrix-streaming@0",
            4, 1, ops.DType.F16,
        ),
        (
            "decode",
            1,
            1024,
            256,
            np.asarray([[0, 1000]], dtype=np.int32),
            "causal_attention.register-partitioned@0",
            4, 1, ops.DType.F16,
        ),
        (
            "prefill",
            19,
            8192,
            256,
            np.asarray([[0, 8000 + index] for index in range(19)], dtype=np.int32),
            "causal_attention.matrix-streaming@0",
            4, 1, ops.DType.F16,
        ),
        (
            "prefill",
            9,
            64,
            24,
            np.asarray([[0, index + 1] for index in range(9)], dtype=np.int32),
            "causal_attention.matrix-streaming@0",
            4, 1, ops.DType.F16,
        ),
        (
            "prefill", 35, 8192, 256,
            np.asarray([[7, 0 if index == 0 else 130 + index] for index in range(35)], dtype=np.int32),
            "causal_attention.matrix-streaming@0", 8, 2, ops.DType.F32,
        ),
        (
            "prefill", 35, 256, 256,
            np.asarray([[7, 215 + index] for index in range(35)], dtype=np.int32),
            "causal_attention.matrix-streaming@0", 6, 2, ops.DType.F32,
        ),
    ),
)
def test_optimized_attention_schedules_match_reference_on_metal(
    mode, rows, capacity, width, visible, expected_name, heads, kv_heads, floating
):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal optimized attention qualification requires MPS")
    rng = np.random.default_rng(45)
    dtype = np.float32 if floating == ops.DType.F32 else np.float16
    query = rng.normal(0, 0.7, (rows, heads, width)).astype(dtype)
    # Distinct queries and KV groups exercise paired and odd head cohorts.
    # FP32 cases include empty partitions, a base offset, an interval ending
    # exactly at physical capacity, and a query-tile tail;
    # their strict tolerance rejects rounding probabilities to a 16-bit dtype.
    history = rng.normal(0, 0.7, (2, capacity, kv_heads, width)).astype(dtype)
    specs = (
        ops.TensorSpec(query.shape, floating),
        ops.TensorSpec(history.shape, floating),
        ops.TensorSpec(visible.shape, ops.DType.I32),
    )
    signature = ops.Signature(
        (
            ops.Argument(specs[0], "query"),
            ops.Argument(specs[1], "history", ops.ValueKind.RESOURCE),
            ops.Argument(specs[2], "visible"),
        )
    )

    def function(q, cache, limits):
        return ops.causal_attention(q, cache, limits, sequence_count=1)

    graph = ops.trace(function, signature)
    expected = ops.evaluate_reference(
        graph,
        {"query": query, "history": history, "visible": visible},
    ).outputs[0]
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 28))
    resources = compiled = execution = None
    try:
        resources = (
            device.upload(specs[0], query.tobytes()),
            device.upload(specs[1], history.tobytes()),
            device.upload(specs[2], visible.tobytes()),
        )
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode=mode),
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
            rtol=3e-5 if floating == ops.DType.F32 else 3e-2,
            atol=3e-6 if floating == ops.DType.F32 else 3e-3,
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
