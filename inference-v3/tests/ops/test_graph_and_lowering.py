import numpy as np
import pytest

import ops
from ops.operation import build_operations
from ops.compiler.lowering import LoweringContext, plan_submissions
from ops.compiler.memory import plan_memory
from ops.compiler.unit import build_unit
from ops.runtime.tilelang import _build_reusable_module
from ops.tensor.graph import prune_dead_nodes


def test_reference_and_pointwise_fusion_preserve_semantics():
    signature = ops.Signature((ops.Argument(ops.TensorSpec((3, 5), ops.DType.F32), "x"),))
    @ops.formula(id="test.pointwise-composition")
    def formula(x):
        return ops.tanh(ops.silu(x) + x)

    ops.operation(formula)(ops.operation_bodies.pointwise)
    graph = ops.trace(formula, signature)
    value = np.arange(15, dtype=np.float32).reshape(3, 5) / 10
    expected = np.tanh((value / (1 + np.exp(-value))) + value)
    np.testing.assert_allclose(ops.evaluate_reference(graph, {"x": value}).outputs[0], expected)

    context = LoweringContext(
        ops.Capabilities(32, 256, 32 * 1024, native_multi_launch=True, partial_binding=True),
        "decode",
        "model",
        "test",
        1 << 20,
    )
    cover = build_operations(graph, context)
    assert len(cover) == 1
    assert cover[0].nodes == frozenset({0, 1, 2})


def test_multi_kernel_cover_becomes_one_tilelang_prim_func():
    signature = ops.Signature((ops.Argument(ops.TensorSpec((2, 4), ops.DType.F32), "x"),))
    graph = ops.trace(lambda x: ops.tanh(ops.rms_norm(x)), signature)
    capabilities = ops.Capabilities(
        32,
        256,
        32 * 1024,
        memory_scopes=frozenset({"global", "shared", "local"}),
        native_multi_launch=True,
        partial_binding=True,
    )
    context = LoweringContext(capabilities, "prefill", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    memory = plan_memory(graph, cover, capabilities)
    units = plan_submissions(graph, cover, capabilities)
    assert len(units) == 1 and units[0].kernel_count == 2
    unit = build_unit(graph, memory, units[0])
    module = _build_reusable_module(unit)
    assert module["main"].attrs["global_symbol"] == "main"
    assert len(unit.calls) == 2


def test_repeated_schedule_shapes_share_private_program_definitions():
    signature = ops.Signature((ops.Argument(ops.TensorSpec((2, 4), ops.DType.F32), "x"),))
    graph = ops.trace(
        lambda x: ops.tanh(ops.rms_norm(ops.tanh(ops.rms_norm(x)))),
        signature,
    )
    capabilities = ops.Capabilities(
        32,
        256,
        32 * 1024,
        memory_scopes=frozenset({"global", "shared", "local"}),
        native_multi_launch=True,
        partial_binding=True,
    )
    context = LoweringContext(capabilities, "prefill", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    memory = plan_memory(graph, cover, capabilities)
    submissions = plan_submissions(graph, cover, capabilities)
    assert len(submissions) == 1
    unit = build_unit(graph, memory, submissions[0])

    module = _build_reusable_module(unit)

    exposed = [
        function
        for function in module.functions.values()
        if function.attrs is not None and function.attrs.get("global_symbol") is not None
    ]
    assert len(unit.calls) == 4
    assert len(module.functions) == 3
    assert len(exposed) == 1


def test_runtime_contract_rejects_python_launch_fallback():
    signature = ops.Signature((ops.Argument(ops.TensorSpec((2, 4), ops.DType.F32)),))
    graph = ops.trace(lambda x: ops.tanh(ops.rms_norm(x)), signature)
    capabilities = ops.Capabilities(
        32,
        256,
        32 * 1024,
        memory_scopes=frozenset({"global", "shared", "local"}),
        partial_binding=True,
    )
    context = LoweringContext(capabilities, "decode", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    with pytest.raises(ValueError, match="native multi-launch"):
        plan_submissions(graph, cover, capabilities)


def test_dead_pure_nodes_are_removed_before_lowering():
    signature = ops.Signature((ops.Argument(ops.TensorSpec((2, 4), ops.DType.F32), "x"),))

    def function(x):
        ops.tanh(x)
        return x

    graph = prune_dead_nodes(ops.trace(function, signature))
    assert not graph.nodes
    assert graph.outputs == graph.inputs


def test_live_multi_output_node_retains_all_emitter_outputs():
    heads = ops.TensorSpec((2, 4, 8), ops.DType.F16)
    positions = ops.TensorSpec((2,), ops.DType.I32)
    signature = ops.Signature(
        (
            ops.Argument(heads, "queries"),
            ops.Argument(heads, "keys"),
            ops.Argument(positions, "positions"),
        )
    )
    graph = prune_dead_nodes(ops.trace(lambda q, k, p: ops.rotary(q, k, p)[0], signature))
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].outputs) == 2
    context = LoweringContext(
        ops.Capabilities(32, 256, 32 * 1024),
        "prefill",
        "model",
        "test",
        1 << 20,
    )
    candidates = build_operations(graph, context)
    primitive = next(candidate for candidate in candidates if candidate.name.startswith("rotary."))
    assert primitive.outputs == graph.nodes[0].outputs


def test_resource_writes_are_versioned_and_aliased():
    history = ops.TensorSpec((2, 8, 2, 4), ops.DType.F32)
    values = ops.TensorSpec((2, 2, 4), ops.DType.F32)
    destinations = ops.TensorSpec((2,), ops.DType.I32)
    signature = ops.Signature(
        (
            ops.Argument(history, "history", ops.ValueKind.RESOURCE),
            ops.Argument(values, "keys"),
            ops.Argument(values, "values"),
            ops.Argument(destinations, "destinations"),
        )
    )
    graph = ops.trace(lambda h, k, v, d: ops.kv_append(h, k, v, d), signature)
    assert graph.nodes[0].effects.writes == ((0, 0, 1),)
    output = graph.values[graph.outputs[0]]
    assert output.resource_id == 0 and output.resource_version == 1


def test_stale_resource_version_cannot_be_reused_after_write():
    history = ops.TensorSpec((2, 8, 2, 4), ops.DType.F32)
    values = ops.TensorSpec((2, 2, 4), ops.DType.F32)
    destinations = ops.TensorSpec((2,), ops.DType.I32)
    signature = ops.Signature(
        (
            ops.Argument(history, "history", ops.ValueKind.RESOURCE),
            ops.Argument(values, "keys"),
            ops.Argument(values, "values"),
            ops.Argument(destinations, "destinations"),
        )
    )

    def stale(h, k, v, d):
        ops.kv_append(h, k, v, d)
        return ops.kv_append(h, k, v, d)

    with pytest.raises(ValueError, match="stale resource version"):
        ops.trace(stale, signature)


def test_pure_resource_consumer_finishes_before_a_later_independent_write():
    vector = ops.TensorSpec((8,), ops.DType.U8)
    extent = ops.TensorSpec((2,), ops.DType.I64)

    def function(state, value, replacement, extent):
        result = (value + value) + state
        following = ops.byte_copy(replacement, state, extent)
        return result, following

    graph = ops.trace(function, ops.Signature((
        ops.Argument(vector, "state", ops.ValueKind.RESOURCE),
        ops.Argument(vector, "value"), ops.Argument(vector, "replacement"),
        ops.Argument(extent, "extent"),
    )))
    read, = (node for node in graph.nodes if node.operation == "add" and graph.resources[0] in node.inputs)
    write, = (node for node in graph.nodes if node.operation == "byte_copy")
    assert read.effects.reads == (graph.value(graph.resources[0]).resource_id,)
    capabilities = ops.Capabilities(32, 256, 32768, native_multi_launch=True, partial_binding=True)
    ordered = build_operations(graph, LoweringContext(capabilities, "prefill", "model", "test", 1 << 20))
    positions = {node: index for index, operation in enumerate(ordered) for node in operation.nodes}
    assert positions[read.id] < positions[write.id]
