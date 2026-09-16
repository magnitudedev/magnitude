import gguf
import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from engine.weights.descriptor import WeightDescriptor
from engine.weights.formats.gguf import Encoding
from engine.weights.tensor_residency import TensorWeights
from ops.compiler.lowering import (
    LoweringContext,
    plan_submissions,
)
from ops.operation import build_operations
from tests.ops.definitions import implement
from tests.ops.test_tilelang_runtime import _packed, _QuantizedFormat

CAPABILITIES = ops.CompilerTarget(
    32,
    256,
    32 * 1024,


    identity="packet-expert-test",
)

GROUPED_CAPABILITIES = ops.CompilerTarget(
    32,
    256,
    32 * 1024,


    identity="grouped-expert-test",
)


def _dense_model(hidden, gate, up, down):
    return ops.linear(ops.silu(ops.linear(hidden, gate)) * ops.linear(hidden, up), down)


_dense_model = implement(_dense_model, ops.operation_bodies.dense_feedforward)


def _encoded_spec(shape):
    return ops.TensorSpec(shape, ops.DType.F16).with_representation(
        ops.Affine(
            ops.Code(8, interpretation=ops.CodeInterpretation.TWOS_COMPLEMENT),
            32,
            ops.DirectCoefficients(ops.DType.F16),
        )
    )


def _mlx_spec(shape):
    return ops.TensorSpec(shape, ops.DType.F16).with_representation(
        ops.Affine(
            ops.Code(4),
            64,
            ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16),
        )
    )


def _routed_shared_model(
    hidden,
    routes,
    scores,
    expert_gate,
    expert_up,
    expert_down,
    shared_gate,
    shared_up,
    shared_down,
    shared_router,
):
    selected = ops.routed_experts(hidden, routes, scores, expert_gate, expert_up, expert_down)
    shared = _dense_model(hidden, shared_gate, shared_up, shared_down)
    coefficient = ops.cast(
        ops.sigmoid(ops.row_dot(hidden, shared_router, output_dtype=ops.DType.F32)),
        hidden.dtype,
    )
    return selected + shared * coefficient


_routed_shared_model = implement(_routed_shared_model, ops.operation_bodies.routed_feedforward)


@pytest.mark.device
def test_direct_dense_swiglu_schedule_matches_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal numerical qualification requires MPS")
    rng = np.random.default_rng(19)
    arrays = (
        rng.normal(0, 0.1, (1, 32)).astype(np.float16),
        rng.normal(0, 0.1, (64, 32)).astype(np.float16),
        rng.normal(0, 0.1, (64, 32)).astype(np.float16),
        rng.normal(0, 0.1, (32, 64)).astype(np.float16),
    )
    specs = tuple(ops.TensorSpec(value.shape, ops.DType.F16) for value in arrays)
    signature = ops.Signature(
        tuple(
            ops.Argument(
                spec,
                name,
                ops.ValueKind.INPUT if index == 0 else ops.ValueKind.CONSTANT,
            )
            for index, (spec, name) in enumerate(
                zip(specs, ("hidden", "gate", "up", "down"), strict=True)
            )
        )
    )
    graph = ops.trace(_dense_model, signature)
    expected = ops.evaluate_reference(
        graph,
        dict(zip(("hidden", "gate", "up", "down"), arrays, strict=True)),
    ).outputs[0]
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    resources = []
    compiled = None
    execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes()) for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = ops.compile(
            _dense_model,
            signature=signature,
            device=device,
            constants={
                name: resource
                for name, resource in zip(("gate", "up", "down"), resources[1:], strict=True)
            },
            options=ops.CompileOptions(mode="decode"),
        )
        assert len(compiled.diagnostics.submissions) == 1
        execution = compiled.submit(resources[0])
        execution.completion.wait()
        np.testing.assert_allclose(
            execution.outputs[0].native.cpu().numpy(), expected, rtol=2e-2, atol=2e-2
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


def test_route_order_and_cutoff_ties_match_qwen_reduction_semantics():
    signature = ops.Signature((ops.Argument(ops.TensorSpec((1, 4), ops.DType.F32), "logits"),))
    graph = ops.trace(
        lambda logits: ops.route_topk(logits, 2, scoring="softmax", normalize=True),
        signature,
    )
    logits = np.asarray([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    indices, scores = ops.evaluate_reference(graph, {"logits": logits}).outputs
    np.testing.assert_array_equal(indices, [[1, 2]])
    np.testing.assert_allclose(scores, [[0.5, 0.5]])


def _graph(rows: int = 1):
    hidden = ops.TensorSpec((rows, 256), ops.DType.F16)
    routes = ops.TensorSpec((rows, 2), ops.DType.I32)
    scores = ops.TensorSpec((rows, 2), ops.DType.F32)
    expert = _encoded_spec((4, 256, 256))
    down = _encoded_spec((4, 256, 256))
    return ops.trace(
        lambda value, indices, weights, gate, up, down_weight: ops.routed_experts(
            value, indices, weights, gate, up, down_weight
        ),
        ops.Signature(
            (
                ops.Argument(hidden, "hidden"),
                ops.Argument(routes, "routes"),
                ops.Argument(scores, "scores"),
                ops.Argument(expert, "gate", ops.ValueKind.CONSTANT),
                ops.Argument(expert, "up", ops.ValueKind.CONSTANT),
                ops.Argument(down, "down", ops.ValueKind.CONSTANT),
            )
        ),
    )


def test_decode_selects_two_stage_packet_expert_lowering_in_one_submission():
    graph = _graph()
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    assert tuple(candidate.name for candidate in cover) == (
        "routed_experts.packet-selected@0",
    )
    submissions = plan_submissions(graph, cover)
    assert len(submissions) == 1
    assert submissions[0].kernel_count == 2


def test_prefill_selects_grouped_expert_pipeline_in_one_submission():
    graph = _graph(rows=8)
    context = LoweringContext(GROUPED_CAPABILITIES, "prefill", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    assert tuple(candidate.name for candidate in cover) == ("routed_experts.grouped@0",)
    submissions = plan_submissions(graph, cover)
    assert len(submissions) == 1
    assert submissions[0].kernel_count == 5


def test_prefill_prepares_rows_and_specializes_mixed_packets_in_one_submission():
    specs = (
        ops.TensorSpec((8, 512), ops.DType.F16),
        ops.TensorSpec((8, 2), ops.DType.I32),
        ops.TensorSpec((8, 2), ops.DType.F32),
        _encoded_spec((4, 256, 512)),
        _encoded_spec((4, 256, 512)),
        _encoded_spec((4, 512, 256)),
        _mlx_spec((512, 512)),
        _mlx_spec((512, 512)),
        _mlx_spec((512, 512)),
        ops.TensorSpec((512,), ops.DType.F16),
    )
    names = (
        "hidden",
        "routes",
        "scores",
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate",
        "shared_up",
        "shared_down",
        "shared_router",
    )
    graph = ops.trace(
        _routed_shared_model,
        ops.Signature(
            tuple(
                ops.Argument(
                    spec,
                    name,
                    ops.ValueKind.INPUT if index < 3 else ops.ValueKind.CONSTANT,
                )
                for index, (name, spec) in enumerate(zip(names, specs, strict=True))
            )
        ),
    )
    context = LoweringContext(GROUPED_CAPABILITIES, "prefill", "model", "test", 1 << 28)
    cover = build_operations(graph, context)

    assert len(cover) == 1
    assert cover[0].name.startswith("routed_experts.grouped@")
    assert cover[0].nodes == frozenset(range(len(graph.nodes)))
    assert cover[0].kernel_count == 7
    assert cover[0].emitter.tile.reduction == 32
    assert cover[0].emitter.specs[6].representation.group == 64
    submissions = plan_submissions(graph, cover)
    assert len(submissions) == 1 and submissions[0].kernel_count == 7


@pytest.mark.parametrize("mode,rows,fused", [("decode", 1, True), ("prefill", 256, False)])
def test_router_fusion_does_not_discard_prefill_matrix_reuse(mode, rows, fused):
    from ops.kernels.routing import RouterTopKRule

    graph = ops.trace(lambda x, w: ops.route_topk(ops.linear(x, w, output_dtype=ops.DType.F32), k=2),
                      ops.Signature((ops.Argument(ops.TensorSpec((rows, 512), ops.DType.F16), "x"),
                                     ops.Argument(ops.TensorSpec((64, 512), ops.DType.F16), "w"))))
    context = LoweringContext(GROUPED_CAPABILITIES, mode, "model", "test", 1 << 28)
    assert bool(RouterTopKRule().build(graph, 0, context)) == fused


def test_prefill_selects_matrix_swiglu_region_in_one_submission():
    specs = (
        ops.TensorSpec((8, 256), ops.DType.F16),
        _encoded_spec((256, 256)),
        _encoded_spec((256, 256)),
        _encoded_spec((256, 256)),
    )
    graph = ops.trace(
        _dense_model,
        ops.Signature(
            tuple(
                ops.Argument(
                    spec,
                    name,
                    ops.ValueKind.INPUT if index == 0 else ops.ValueKind.CONSTANT,
                )
                for index, (spec, name) in enumerate(
                    zip(specs, ("hidden", "gate", "up", "down"), strict=True)
                )
            )
        ),
    )
    context = LoweringContext(GROUPED_CAPABILITIES, "prefill", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    assert tuple(candidate.name for candidate in cover) == (
        "dense_swiglu.packet-prefill@0:4",
    )
    submissions = plan_submissions(graph, cover)
    assert len(submissions) == 1 and submissions[0].kernel_count == 2


@pytest.mark.device
def test_matrix_prefill_swiglu_matches_reference_on_metal():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal matrix SwiGLU qualification requires MPS")
    rng = np.random.default_rng(90)
    arrays = (
        rng.normal(0, 0.1, (8, 32)).astype(np.float16),
        rng.normal(0, 0.1, (64, 32)).astype(np.float16),
        rng.normal(0, 0.1, (64, 32)).astype(np.float16),
        rng.normal(0, 0.1, (32, 64)).astype(np.float16),
    )
    specs = tuple(ops.TensorSpec(value.shape, ops.DType.F16) for value in arrays)
    signature = ops.Signature(
        tuple(
            ops.Argument(
                spec,
                name,
                ops.ValueKind.INPUT if index == 0 else ops.ValueKind.CONSTANT,
            )
            for index, (spec, name) in enumerate(
                zip(specs, ("hidden", "gate", "up", "down"), strict=True)
            )
        )
    )
    graph = ops.trace(_dense_model, signature)
    expected = ops.evaluate_reference(
        graph, dict(zip(("hidden", "gate", "up", "down"), arrays, strict=True))
    ).outputs[0]
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes()) for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = ops.compile(
            _dense_model,
            signature=signature,
            device=device,
            constants={
                name: resource
                for name, resource in zip(("gate", "up", "down"), resources[1:], strict=True)
            },
            options=ops.CompileOptions(mode="prefill"),
        )
        assert len(compiled.diagnostics.submissions) == 1
        execution = compiled.submit(resources[0])
        execution.completion.wait()
        np.testing.assert_allclose(
            execution.outputs[0].native.cpu().numpy(), expected, rtol=3e-2, atol=3e-2
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
@pytest.mark.parametrize("rows", [8, 33])
def test_grouped_prefill_consumes_quantized_experts_without_materialization(rows):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal encoded expert qualification requires MPS")
    experts, intermediate, width, selected = 8, 256, 256, 2
    encoding = Encoding.Q8_0
    packed = (
        _packed(encoding, experts * intermediate, width),
        _packed(encoding, experts * intermediate, width),
        _packed(encoding, experts * width, intermediate),
    )
    rng = np.random.default_rng(93)
    hidden = rng.normal(0, 0.1, (rows, width)).astype(np.float16)
    # Leave most of the statically provisioned expert blocks inactive. This
    # covers the production capacity path where unused blocks must perform no
    # packed matrix reduction work and must not affect the result.
    routes = np.asarray([[0, experts - 1] for _ in range(rows)], dtype=np.int32)
    scores = np.full((rows, selected), 0.5, dtype=np.float32)
    decoded = (
        gguf.dequantize(
            packed[0].reshape(experts * intermediate, -1),
            gguf.GGMLQuantizationType(encoding),
        ).reshape(experts, intermediate, width),
        gguf.dequantize(
            packed[1].reshape(experts * intermediate, -1),
            gguf.GGMLQuantizationType(encoding),
        ).reshape(experts, intermediate, width),
        gguf.dequantize(
            packed[2].reshape(experts * width, -1),
            gguf.GGMLQuantizationType(encoding),
        ).reshape(experts, width, intermediate),
    )
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=8 << 20))
    owners = [
        TensorWeights(_QuantizedFormat(value.tobytes(), encoding), device) for value in packed
    ]
    weights = [
        owners[0].bind(
            WeightDescriptor(name="gate", shape=(experts, intermediate, width)),
            ops.DType.F16,
        ),
        owners[1].bind(
            WeightDescriptor(name="up", shape=(experts, intermediate, width)),
            ops.DType.F16,
        ),
        owners[2].bind(
            WeightDescriptor(name="down", shape=(experts, width, intermediate)),
            ops.DType.F16,
        ),
    ]
    inputs = [
        device.upload(ops.TensorSpec(hidden.shape, ops.DType.F16), hidden.tobytes()),
        device.upload(ops.TensorSpec(routes.shape, ops.DType.I32), routes.tobytes()),
        device.upload(ops.TensorSpec(scores.shape, ops.DType.F32), scores.tobytes()),
    ]
    signature = ops.Signature(
        (
            ops.Argument(inputs[0].spec, "hidden"),
            ops.Argument(inputs[1].spec, "routes"),
            ops.Argument(inputs[2].spec, "scores"),
            ops.Argument(weights[0].spec, "gate", ops.ValueKind.CONSTANT),
            ops.Argument(weights[1].spec, "up", ops.ValueKind.CONSTANT),
            ops.Argument(weights[2].spec, "down", ops.ValueKind.CONSTANT),
        )
    )
    graph = ops.trace(ops.routed_experts, signature)
    expected, = ops.evaluate_reference(graph, dict(zip(
        (*graph.inputs, *graph.constants), (hidden, routes, scores, *decoded), strict=True,
    ))).outputs
    compiled = execution = None
    try:
        compiled = ops.compile(
            ops.routed_experts,
            signature=signature,
            device=device,
            constants={"gate": weights[0], "up": weights[1], "down": weights[2]},
            options=ops.CompileOptions(mode="prefill"),
        )
        assert compiled.diagnostics.submissions == (("routed_experts.grouped@0",),)
        assert all(value.spec.representation is not None for value in weights)
        execution = compiled.submit(*inputs)
        execution.completion.wait()
        np.testing.assert_allclose(
            execution.outputs[0].native.cpu().numpy(),
            expected,
            rtol=3e-2,
            atol=5e-2,
        )
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(inputs):
            resource.close()
        for owner in reversed(owners):
            owner.close()
        device.close()


@pytest.mark.device
@pytest.mark.parametrize("scoring,normalize", [("softmax", True), ("sigmoid", False)])
def test_parallel_router_preserves_scores_and_cutoff_ties(scoring, normalize):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(812)
    hidden = rng.normal(0, .2, (2, 2048)).astype(np.float16)
    hidden[0] = 0  # All 256 experts tie; preserve index and output ordering.
    router = rng.normal(0, .2, (256, 2048)).astype(np.float32)
    specs = (ops.TensorSpec(hidden.shape, ops.DType.F16),
             ops.TensorSpec(router.shape, ops.DType.F32))
    signature = ops.Signature((ops.Argument(specs[0], "hidden"), ops.Argument(specs[1], "router")))

    def function(x, w):
        return ops.route_topk(ops.linear(x, w, output_dtype=ops.DType.F32),
                              k=8, scoring=scoring, normalize=normalize)

    function = implement(function, ops.operation_bodies.routed_feedforward)
    graph = ops.trace(function, signature)
    expected = ops.evaluate_reference(graph, {"hidden": hidden, "router": router}).outputs
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=32 << 20))
    resources = []
    compiled = execution = None
    try:
        for spec, value in zip(specs, (hidden, router), strict=True):
            resources.append(device.upload(spec, value.tobytes()))
        compiled = ops.compile(function, signature=signature, device=device, constants={},
                               options=ops.CompileOptions(mode="decode"))
        assert compiled.diagnostics.dispatches == 2
        assert "route_topk.parallel-router" in str(compiled.diagnostics.submissions)
        execution = compiled.submit(*resources)
        execution.completion.wait()
        np.testing.assert_array_equal(execution.outputs[0].native.cpu().numpy(), expected[0])
        np.testing.assert_allclose(execution.outputs[1].native.cpu().numpy(), expected[1],
                                   rtol=3e-5, atol=3e-6)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()
