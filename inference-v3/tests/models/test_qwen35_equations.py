import magnitensor as mt
from magnitensor.compiler.lowering import (
    LoweringContext,
    plan_submissions,
    select_cover,
)
from magnitude_engine.models.qwen35.equations import (
    DenseFeedForwardTensors,
    RoutedFeedForwardTensors,
    dense_feedforward,
    routed_feedforward,
)

CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    memory_scopes=frozenset({"global", "shared", "local"}),
    native_multi_launch=True,
    partial_binding=True,
    fingerprint="qwen-equations-test",
)


def _constant(shape, name):
    return mt.Argument(mt.TensorSpec(shape, mt.DType.F16), name, mt.ValueKind.CONSTANT)


def test_dense_feedforward_is_model_composition_not_an_execution_object():
    signature = mt.Signature(
        (
            mt.Argument(mt.TensorSpec((1, 32), mt.DType.F16), "hidden"),
            _constant((64, 32), "gate"),
            _constant((64, 32), "up"),
            _constant((32, 64), "down"),
        )
    )
    graph = mt.trace(
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
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    assert tuple(candidate.name for candidate in cover.candidates) == ("dense_swiglu.direct@0:4",)
    submissions = plan_submissions(graph, cover, CAPABILITIES)
    assert len(submissions) == 1
    assert submissions[0].kernel_count == 2


def test_routed_feedforward_lowers_as_one_maximal_decode_submission():
    signature = mt.Signature(
        (
            mt.Argument(mt.TensorSpec((1, 32), mt.DType.F16), "hidden"),
            _constant((4, 32), "router"),
            mt.Argument(
                mt.TensorSpec((32,), mt.DType.F32),
                "shared_router",
                mt.ValueKind.CONSTANT,
            ),
            _constant((4, 64, 32), "expert_gate"),
            _constant((4, 64, 32), "expert_up"),
            _constant((4, 32, 64), "expert_down"),
            _constant((64, 32), "shared_gate"),
            _constant((64, 32), "shared_up"),
            _constant((32, 64), "shared_down"),
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

    graph = mt.trace(model, signature)
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    assert any(
        candidate.name.startswith("routed_experts.direct@") for candidate in cover.candidates
    )
    submissions = plan_submissions(graph, cover, CAPABILITIES)
    assert len(submissions) == 1
    assert submissions[0].kernel_count > 1
