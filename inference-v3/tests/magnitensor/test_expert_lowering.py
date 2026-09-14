import gguf
import numpy as np
import pytest
import torch

import magnitensor as mt
from magnitensor.compiler.lowering import (
    LoweringContext,
    plan_submissions,
    select_cover,
)
from magnitude_engine.weights.descriptor import WeightDescriptor
from magnitude_engine.weights.formats.gguf import Encoding
from magnitude_engine.weights.tensor_residency import TensorWeights
from tests.magnitensor.test_tilelang_runtime import _packed, _QuantizedFormat

CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    memory_scopes=frozenset({"global", "shared", "local"}),
    native_multi_launch=True,
    partial_binding=True,
    fingerprint="packet-expert-test",
)

GROUPED_CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    matrix_instructions=(mt.MatrixInstruction(8, 8, 8, mt.DType.F16, mt.DType.F32),),
    memory_scopes=frozenset({"global", "shared", "local"}),
    atomics=frozenset({mt.DType.I32}),
    features=frozenset({"gemm.runtime_valid_m"}),
    native_multi_launch=True,
    partial_binding=True,
    fingerprint="grouped-expert-test",
)


def _dense_model(hidden, gate, up, down):
    return mt.linear(mt.silu(mt.linear(hidden, gate)) * mt.linear(hidden, up), down)


def _encoded_spec(shape):
    return mt.TensorSpec(shape, mt.DType.F16).with_representation(
        mt.Affine(
            mt.Code(8, interpretation=mt.CodeInterpretation.TWOS_COMPLEMENT),
            32,
            mt.DirectCoefficients(mt.DType.F16),
        )
    )


def _mlx_spec(shape):
    return mt.TensorSpec(shape, mt.DType.F16).with_representation(
        mt.Affine(
            mt.Code(4),
            64,
            mt.DirectCoefficients(mt.DType.BF16, mt.DType.BF16),
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
    selected = mt.routed_experts(hidden, routes, scores, expert_gate, expert_up, expert_down)
    shared = _dense_model(hidden, shared_gate, shared_up, shared_down)
    coefficient = mt.cast(
        mt.sigmoid(mt.row_dot(hidden, shared_router, output_dtype=mt.DType.F32)),
        hidden.dtype,
    )
    return selected + shared * coefficient


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
    specs = tuple(mt.TensorSpec(value.shape, mt.DType.F16) for value in arrays)
    signature = mt.Signature(
        tuple(
            mt.Argument(
                spec,
                name,
                mt.ValueKind.INPUT if index == 0 else mt.ValueKind.CONSTANT,
            )
            for index, (spec, name) in enumerate(
                zip(specs, ("hidden", "gate", "up", "down"), strict=True)
            )
        )
    )
    graph = mt.trace(_dense_model, signature)
    expected = mt.evaluate_reference(
        graph,
        dict(zip(("hidden", "gate", "up", "down"), arrays, strict=True)),
    ).outputs[0]
    device = mt.device("metal", budget_bytes=1 << 20)
    resources = []
    compiled = None
    execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes()) for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = mt.compile(
            _dense_model,
            signature=signature,
            device=device,
            constants={
                name: resource
                for name, resource in zip(("gate", "up", "down"), resources[1:], strict=True)
            },
            options=mt.CompileOptions(mode="decode"),
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
    signature = mt.Signature((mt.Argument(mt.TensorSpec((1, 4), mt.DType.F32), "logits"),))
    graph = mt.trace(
        lambda logits: mt.route_topk(logits, 2, scoring="softmax", normalize=True),
        signature,
    )
    logits = np.asarray([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    indices, scores = mt.evaluate_reference(graph, {"logits": logits}).outputs
    np.testing.assert_array_equal(indices, [[1, 2]])
    np.testing.assert_allclose(scores, [[0.5, 0.5]])


def _graph(rows: int = 1):
    hidden = mt.TensorSpec((rows, 256), mt.DType.F16)
    routes = mt.TensorSpec((rows, 2), mt.DType.I32)
    scores = mt.TensorSpec((rows, 2), mt.DType.F32)
    expert = _encoded_spec((4, 256, 256))
    down = _encoded_spec((4, 256, 256))
    return mt.trace(
        lambda value, indices, weights, gate, up, down_weight: mt.routed_experts(
            value, indices, weights, gate, up, down_weight
        ),
        mt.Signature(
            (
                mt.Argument(hidden, "hidden"),
                mt.Argument(routes, "routes"),
                mt.Argument(scores, "scores"),
                mt.Argument(expert, "gate", mt.ValueKind.CONSTANT),
                mt.Argument(expert, "up", mt.ValueKind.CONSTANT),
                mt.Argument(down, "down", mt.ValueKind.CONSTANT),
            )
        ),
    )


def test_decode_selects_two_stage_packet_expert_lowering_in_one_submission():
    graph = _graph()
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    assert tuple(candidate.name for candidate in cover.candidates) == (
        "routed_experts.packet-selected@0",
    )
    submissions = plan_submissions(graph, cover, CAPABILITIES)
    assert len(submissions) == 1
    assert submissions[0].kernel_count == 2


def test_prefill_selects_grouped_expert_pipeline_in_one_submission():
    graph = _graph(rows=8)
    context = LoweringContext(GROUPED_CAPABILITIES, "prefill", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    assert tuple(candidate.name for candidate in cover.candidates) == ("routed_experts.grouped@0",)
    submissions = plan_submissions(graph, cover, GROUPED_CAPABILITIES)
    assert len(submissions) == 1
    assert submissions[0].kernel_count == 4


def test_prefill_combines_mixed_packet_routed_and_shared_experts_in_four_kernels():
    specs = (
        mt.TensorSpec((8, 512), mt.DType.F16),
        mt.TensorSpec((8, 2), mt.DType.I32),
        mt.TensorSpec((8, 2), mt.DType.F32),
        _encoded_spec((4, 256, 512)),
        _encoded_spec((4, 256, 512)),
        _encoded_spec((4, 512, 256)),
        _mlx_spec((512, 512)),
        _mlx_spec((512, 512)),
        _mlx_spec((512, 512)),
        mt.TensorSpec((512,), mt.DType.F16),
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
    graph = mt.trace(
        _routed_shared_model,
        mt.Signature(
            tuple(
                mt.Argument(
                    spec,
                    name,
                    mt.ValueKind.INPUT if index < 3 else mt.ValueKind.CONSTANT,
                )
                for index, (name, spec) in enumerate(zip(names, specs, strict=True))
            )
        ),
    )
    context = LoweringContext(GROUPED_CAPABILITIES, "prefill", "model", "test", 1 << 28)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))

    assert len(cover.candidates) == 1
    assert cover.candidates[0].name.startswith("routed_experts.grouped@")
    assert cover.candidates[0].nodes == frozenset(range(len(graph.nodes)))
    assert cover.candidates[0].kernel_count == 4


def test_prefill_selects_matrix_swiglu_region_in_one_submission():
    specs = (
        mt.TensorSpec((8, 256), mt.DType.F16),
        _encoded_spec((256, 256)),
        _encoded_spec((256, 256)),
        _encoded_spec((256, 256)),
    )
    graph = mt.trace(
        _dense_model,
        mt.Signature(
            tuple(
                mt.Argument(
                    spec,
                    name,
                    mt.ValueKind.INPUT if index == 0 else mt.ValueKind.CONSTANT,
                )
                for index, (spec, name) in enumerate(
                    zip(specs, ("hidden", "gate", "up", "down"), strict=True)
                )
            )
        ),
    )
    context = LoweringContext(GROUPED_CAPABILITIES, "prefill", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    assert tuple(candidate.name for candidate in cover.candidates) == (
        "dense_swiglu.packet-prefill@0:4",
    )
    submissions = plan_submissions(graph, cover, GROUPED_CAPABILITIES)
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
    specs = tuple(mt.TensorSpec(value.shape, mt.DType.F16) for value in arrays)
    signature = mt.Signature(
        tuple(
            mt.Argument(
                spec,
                name,
                mt.ValueKind.INPUT if index == 0 else mt.ValueKind.CONSTANT,
            )
            for index, (spec, name) in enumerate(
                zip(specs, ("hidden", "gate", "up", "down"), strict=True)
            )
        )
    )
    graph = mt.trace(_dense_model, signature)
    expected = mt.evaluate_reference(
        graph, dict(zip(("hidden", "gate", "up", "down"), arrays, strict=True))
    ).outputs[0]
    device = mt.device("metal", budget_bytes=1 << 20)
    resources = []
    compiled = execution = None
    try:
        resources = [
            device.upload(spec, value.tobytes()) for spec, value in zip(specs, arrays, strict=True)
        ]
        compiled = mt.compile(
            _dense_model,
            signature=signature,
            device=device,
            constants={
                name: resource
                for name, resource in zip(("gate", "up", "down"), resources[1:], strict=True)
            },
            options=mt.CompileOptions(mode="prefill"),
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
def test_grouped_prefill_consumes_quantized_experts_without_materialization():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal encoded expert qualification requires MPS")
    experts, intermediate, width, rows, selected = 8, 256, 256, 8, 2
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
    routes = np.asarray([[0, 1] for _ in range(rows)], dtype=np.int32)
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
    expected = np.zeros((rows, width), dtype=np.float32)
    for row in range(rows):
        for rank in range(selected):
            expert = routes[row, rank]
            gate_value = hidden[row].astype(np.float32) @ decoded[0][expert].astype(np.float32).T
            up_value = hidden[row].astype(np.float32) @ decoded[1][expert].astype(np.float32).T
            activated = gate_value / (1 + np.exp(-gate_value)) * up_value
            expected[row] += (activated @ decoded[2][expert].astype(np.float32).T) * scores[
                row, rank
            ]
    device = mt.device("metal", budget_bytes=8 << 20)
    owners = [
        TensorWeights(_QuantizedFormat(value.tobytes(), encoding), device) for value in packed
    ]
    weights = [
        owners[0].resident(
            WeightDescriptor(name="gate", shape=(experts, intermediate, width)),
            mt.DType.F16,
        ),
        owners[1].resident(
            WeightDescriptor(name="up", shape=(experts, intermediate, width)),
            mt.DType.F16,
        ),
        owners[2].resident(
            WeightDescriptor(name="down", shape=(experts, width, intermediate)),
            mt.DType.F16,
        ),
    ]
    inputs = [
        device.upload(mt.TensorSpec(hidden.shape, mt.DType.F16), hidden.tobytes()),
        device.upload(mt.TensorSpec(routes.shape, mt.DType.I32), routes.tobytes()),
        device.upload(mt.TensorSpec(scores.shape, mt.DType.F32), scores.tobytes()),
    ]
    signature = mt.Signature(
        (
            mt.Argument(inputs[0].spec, "hidden"),
            mt.Argument(inputs[1].spec, "routes"),
            mt.Argument(inputs[2].spec, "scores"),
            mt.Argument(weights[0].spec, "gate", mt.ValueKind.CONSTANT),
            mt.Argument(weights[1].spec, "up", mt.ValueKind.CONSTANT),
            mt.Argument(weights[2].spec, "down", mt.ValueKind.CONSTANT),
        )
    )
    compiled = execution = None
    try:
        compiled = mt.compile(
            lambda value, indices, probabilities, gate, up, down: mt.routed_experts(
                value, indices, probabilities, gate, up, down
            ),
            signature=signature,
            device=device,
            constants={"gate": weights[0], "up": weights[1], "down": weights[2]},
            options=mt.CompileOptions(mode="prefill"),
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
