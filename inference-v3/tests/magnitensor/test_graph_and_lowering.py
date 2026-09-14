import numpy as np
import pytest

import magnitensor as mt
from magnitensor.compiler.lowering import LoweringContext, plan_submissions, select_cover
from magnitensor.compiler.memory import plan_memory
from magnitensor.compiler.unit import build_unit
from magnitensor.runtime.tilelang import _build_prim_func, _build_reusable_module
from magnitensor.tensor.graph import prune_dead_nodes


def test_reference_and_pointwise_fusion_preserve_semantics():
    signature = mt.Signature((mt.Argument(mt.TensorSpec((3, 5), mt.DType.F32), "x"),))
    graph = mt.trace(lambda x: mt.tanh(mt.silu(x) + x), signature)
    value = np.arange(15, dtype=np.float32).reshape(3, 5) / 10
    expected = np.tanh((value / (1 + np.exp(-value))) + value)
    np.testing.assert_allclose(mt.evaluate_reference(graph, {"x": value}).outputs[0], expected)

    context = LoweringContext(
        mt.Capabilities(32, 256, 32 * 1024, native_multi_launch=True, partial_binding=True),
        "decode",
        "model",
        "test",
        1 << 20,
    )
    candidates = mt.lowerings.enumerate(graph, context)
    cover = select_cover(graph, candidates)
    assert len(cover.candidates) == 1
    assert cover.candidates[0].nodes == frozenset({0, 1, 2})


def test_multi_kernel_cover_becomes_one_tilelang_prim_func():
    signature = mt.Signature((mt.Argument(mt.TensorSpec((2, 4), mt.DType.F32), "x"),))
    graph = mt.trace(lambda x: mt.tanh(mt.rms_norm(x)), signature)
    capabilities = mt.Capabilities(
        32,
        256,
        32 * 1024,
        memory_scopes=frozenset({"global", "shared", "local"}),
        native_multi_launch=True,
        partial_binding=True,
    )
    context = LoweringContext(capabilities, "prefill", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    memory = plan_memory(graph, cover, capabilities)
    units = plan_submissions(graph, cover, capabilities)
    assert len(units) == 1 and units[0].kernel_count == 2
    unit = build_unit(graph, memory, units[0])
    prim_func = _build_prim_func(unit)
    assert prim_func.attrs is not None
    assert prim_func.attrs["global_symbol"] == unit.name
    assert len(unit.calls) == 2


def test_repeated_schedule_shapes_share_private_program_definitions():
    signature = mt.Signature((mt.Argument(mt.TensorSpec((2, 4), mt.DType.F32), "x"),))
    graph = mt.trace(
        lambda x: mt.tanh(mt.rms_norm(mt.tanh(mt.rms_norm(x)))),
        signature,
    )
    capabilities = mt.Capabilities(
        32,
        256,
        32 * 1024,
        memory_scopes=frozenset({"global", "shared", "local"}),
        native_multi_launch=True,
        partial_binding=True,
    )
    context = LoweringContext(capabilities, "prefill", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
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
    signature = mt.Signature((mt.Argument(mt.TensorSpec((2, 4), mt.DType.F32)),))
    graph = mt.trace(lambda x: mt.tanh(mt.rms_norm(x)), signature)
    capabilities = mt.Capabilities(
        32,
        256,
        32 * 1024,
        memory_scopes=frozenset({"global", "shared", "local"}),
        partial_binding=True,
    )
    context = LoweringContext(capabilities, "decode", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    with pytest.raises(ValueError, match="native multi-launch"):
        plan_submissions(graph, cover, capabilities)


def test_dead_pure_nodes_are_removed_before_lowering():
    signature = mt.Signature((mt.Argument(mt.TensorSpec((2, 4), mt.DType.F32), "x"),))

    def function(x):
        mt.tanh(x)
        return x

    graph = prune_dead_nodes(mt.trace(function, signature))
    assert not graph.nodes
    assert graph.outputs == graph.inputs


def test_live_multi_output_node_retains_all_emitter_outputs():
    heads = mt.TensorSpec((2, 4, 8), mt.DType.F16)
    positions = mt.TensorSpec((2,), mt.DType.I32)
    signature = mt.Signature(
        (
            mt.Argument(heads, "queries"),
            mt.Argument(heads, "keys"),
            mt.Argument(positions, "positions"),
        )
    )
    graph = prune_dead_nodes(mt.trace(lambda q, k, p: mt.rotary(q, k, p)[0], signature))
    assert len(graph.nodes) == 1
    assert len(graph.nodes[0].outputs) == 2
    context = LoweringContext(
        mt.Capabilities(32, 256, 32 * 1024),
        "prefill",
        "model",
        "test",
        1 << 20,
    )
    candidates = mt.lowerings.enumerate(graph, context)
    primitive = next(candidate for candidate in candidates if candidate.name.startswith("rotary."))
    assert primitive.outputs == graph.nodes[0].outputs


def test_resource_writes_are_versioned_and_aliased():
    history = mt.TensorSpec((2, 8, 2, 4), mt.DType.F32)
    values = mt.TensorSpec((2, 2, 4), mt.DType.F32)
    destinations = mt.TensorSpec((2,), mt.DType.I32)
    signature = mt.Signature(
        (
            mt.Argument(history, "history", mt.ValueKind.RESOURCE),
            mt.Argument(values, "keys"),
            mt.Argument(values, "values"),
            mt.Argument(destinations, "destinations"),
        )
    )
    graph = mt.trace(lambda h, k, v, d: mt.kv_append(h, k, v, d), signature)
    assert graph.nodes[0].effects.writes == ((0, 0, 1),)
    output = graph.values[graph.outputs[0]]
    assert output.resource_id == 0 and output.resource_version == 1


def test_stale_resource_version_cannot_be_reused_after_write():
    history = mt.TensorSpec((2, 8, 2, 4), mt.DType.F32)
    values = mt.TensorSpec((2, 2, 4), mt.DType.F32)
    destinations = mt.TensorSpec((2,), mt.DType.I32)
    signature = mt.Signature(
        (
            mt.Argument(history, "history", mt.ValueKind.RESOURCE),
            mt.Argument(values, "keys"),
            mt.Argument(values, "values"),
            mt.Argument(destinations, "destinations"),
        )
    )

    def stale(h, k, v, d):
        mt.kv_append(h, k, v, d)
        return mt.kv_append(h, k, v, d)

    with pytest.raises(ValueError, match="stale resource version"):
        mt.trace(stale, signature)
