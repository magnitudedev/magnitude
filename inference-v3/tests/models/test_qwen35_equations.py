import ops
from engine.models.qwen35 import operations as _operations
from ops.operation import build_operations
from ops.compiler.lowering import (
    LoweringContext,
    plan_submissions,
)
from engine.models.qwen35.equations import (
    DenseFeedForwardTensors,
    RoutedFeedForwardTensors,
    dense_feedforward,
    routed_feedforward,
)

CAPABILITIES = ops.CompilerTarget(
    32,
    256,
    32 * 1024,


    identity="qwen-equations-test",
)


def _constant(shape, name):
    return ops.Argument(ops.TensorSpec(shape, ops.DType.F16), name, ops.ValueKind.CONSTANT)


def _packed_constant(shape, name):
    spec = ops.TensorSpec(shape, ops.DType.F16).with_representation(
        ops.Affine(
            ops.Code(4),
            64,
            ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16),
        )
    )
    return ops.Argument(spec, name, ops.ValueKind.CONSTANT)


def test_dense_feedforward_is_model_composition_not_an_execution_object():
    signature = ops.Signature(
        (
            ops.Argument(ops.TensorSpec((1, 512), ops.DType.F16), "hidden"),
            _packed_constant((512, 512), "gate"),
            _packed_constant((512, 512), "up"),
            _packed_constant((512, 512), "down"),
        )
    )
    graph = ops.trace(
        lambda hidden, gate, up, down: dense_feedforward(
            hidden, DenseFeedForwardTensors(gate, up, down)
        ),
        signature,
    )
    assert tuple(node.operation for node in graph.nodes) == (
        "linear",
        "linear",
        "silu",
        "multiply",
        "linear",
    )
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    assert tuple(candidate.name for candidate in cover) == (
        "dense_swiglu.packet-decode@0:4",
    )
    submissions = plan_submissions(graph, cover)
    assert len(submissions) == 1
    assert submissions[0].kernel_count == 2


def test_routed_feedforward_lowers_as_one_maximal_decode_submission():
    signature = ops.Signature(
        (
            ops.Argument(ops.TensorSpec((1, 512), ops.DType.F16), "hidden"),
            _constant((4, 512), "router"),
            ops.Argument(
                ops.TensorSpec((512,), ops.DType.F32),
                "shared_router",
                ops.ValueKind.CONSTANT,
            ),
            _packed_constant((4, 512, 512), "expert_gate"),
            _packed_constant((4, 512, 512), "expert_up"),
            _packed_constant((4, 512, 512), "expert_down"),
            _packed_constant((512, 512), "shared_gate"),
            _packed_constant((512, 512), "shared_up"),
            _packed_constant((512, 512), "shared_down"),
        )
    )

    def model(
        hidden,
        router,
        shared_router,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
    ):
        return routed_feedforward(
            hidden,
            RoutedFeedForwardTensors(
                router,
                shared_router,
                expert_gate,
                expert_up,
                expert_down,
                DenseFeedForwardTensors(shared_gate, shared_up, shared_down),
                selected=2,
                normalize_selected=True,
            ),
        )

    graph = ops.trace(model, signature)
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    names = tuple(candidate.name.split("@", 1)[0] for candidate in cover)
    assert names == ("route_topk.fused-router", "routed_experts.packet-shared")
    submissions = plan_submissions(graph, cover)
    assert len(submissions) == 1
    assert submissions[0].kernel_count == 3
