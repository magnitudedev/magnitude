"""Production numerical declarations survive device-free analytical publication."""

from types import SimpleNamespace

import numpy as np
from formula_performance.records import Publication

import ops
from ops.performance.publication import capture, manifest
from ops.performance.traffic import boundary_traffic, boundary_traffic_bound
from ops.runtime.observation import KernelActivity, KernelObservation


@ops.formula(id="test.model.performance", metric="tokens", rows="x")
def model(x, w):
    return ops.silu(ops.linear(x, w))


def test_production_graph_publishes_declared_metrics_and_immutable_math():
    g = ops.trace(
        model,
        ops.Signature(
            (
                ops.Argument(ops.TensorSpec((2, 4), ops.DType.F32)),
                ops.Argument(ops.TensorSpec((8, 4), ops.DType.F32)),
            )
        ),
    )
    m = manifest(g)
    assert [f.primary.name for f in m.formulas] == ["tokens", "floating-work", "output-elements"]
    assert m.formulas[0].primary.expression.evaluate(m.parameters) == 2
    assert m.numerical_graph["nodes"][0]["operation"] == "linear"
    assert Publication.model_validate_json(
        Publication(manifests=(m,)).model_dump_json()
    ).manifests == (m,)
    target = ops.FormulaTree(g).roots[0].children[0]
    isolated = manifest(target.isolate().graph)
    # Isolating an occurrence preserves its original index, not its array offset.
    activity = KernelActivity(
        "linear",
        100,
        10,
        110,
        0,
        (target.call.occurrence,),
        target.call.occurrence,
        0,
        isolated.graph,
    )
    observed = SimpleNamespace(
        kernels=KernelObservation("gpu", (activity,), "compiled-order-and-symbols")
    )
    c = capture(isolated, observed, capture_id="native")
    assert c.coverage == "complete"
    assert c.regions[0].owners == ("",)


def test_partial_binding_preserves_a_proven_traffic_bound_without_inventing_indices():
    @ops.formula(id="test.indexed", metric="tokens", rows="indices")
    def indexed(indices, table, w):
        return ops.linear(ops.embedding(indices, table), w)

    g = ops.trace(
        indexed,
        ops.Signature(
            (
                ops.Argument(ops.TensorSpec((2,), ops.DType.I32)),
                ops.Argument(ops.TensorSpec((32, 4), ops.DType.F32)),
                ops.Argument(ops.TensorSpec((8, 4), ops.DType.F32)),
            )
        ),
    )
    lower, missing = boundary_traffic_bound(g, {})
    exact = boundary_traffic(g, {g.inputs[0]: np.array([1, 2], dtype=np.int32)})
    assert missing and 0 < lower < exact
    m = manifest(g)
    assert m.unresolved
    assert m.parameters["boundary:"] == lower


def test_partially_bound_attention_work_retains_symbolic_interval():
    from ops.performance.semantics import persistent_attention

    inputs = (
        ops.TensorSpec((1, 2, 4), ops.DType.F32),
        ops.TensorSpec((16, 1, 8), ops.DType.F16),
        ops.TensorSpec((1, 1, 4), ops.DType.F32),
        ops.TensorSpec((1, 1, 4), ops.DType.F32),
        ops.TensorSpec((1, 4), ops.DType.I32),
    )
    result = persistent_attention(inputs, {}, (), values=(None, None, None, None, None))
    assert result.matrix.lower == 0 and result.matrix.upper > 0
    assert result.issues
    concrete = persistent_attention(
        inputs, {}, (), values=(None, None, None, None, np.array([[0, 4, 0, 1]]))
    )
    assert concrete.matrix.fixed and concrete.matrix.lower > 0


def test_dynamic_dispatches_keep_enclosing_origins_and_exact_order():
    from ops.runtime.observation import RuntimeCapture, RuntimeRecorder

    g = ops.trace(
        model,
        ops.Signature(
            (
                ops.Argument(ops.TensorSpec((2, 4), ops.DType.F32)),
                ops.Argument(ops.TensorSpec((8, 4), ops.DType.F32)),
            )
        ),
    )
    target = ops.FormulaTree(g).roots[0].children[0]
    operation = SimpleNamespace(
        nodes=target.call.nodes, definition=SimpleNamespace(name="linear"), kernel_count=1
    )
    unit = SimpleNamespace(calls=(SimpleNamespace(operation=operation),))
    recorder = RuntimeRecorder(SimpleNamespace())
    capture = RuntimeCapture(recorder)
    capture._native = SimpleNamespace(clock="gpu")
    recorder._active = capture
    invocation = recorder.invocation()
    # A runtime-selected source stage followed by a nested conversion program.
    with recorder.scope(g, unit, invocation):
        recorder.stage("gather")
        inner = target.isolate().graph
        recorder.dispatch(inner, unit, recorder.invocation())
    recorder.dispatch(g, unit, invocation)
    result = capture._attribute(
        tuple(
            KernelActivity(name, 10, i * 10, (i + 1) * 10, i)
            for i, name in enumerate(("gather", "linear", "linear"))
        )
    )
    assert result.attribution == "compiled-order-and-symbols"
    assert all(
        a.graph == g.fingerprint
        and a.owner == target.call.occurrence
        and a.invocation == invocation
        for a in result.activities
    )
    assert capture._attribute(result.activities[:-1]).attribution.startswith("unavailable")


def test_bounded_reference_loads_weights_at_use_and_releases_after_last_consumer():
    import weakref

    from ops.tensor.primitive import evaluate_reference

    @ops.formula(id="test.lazy.reference", metric="tokens", rows="x")
    def sequential_weights(x, first, second):
        return ops.linear(ops.linear(x, first), second)

    spec = ops.TensorSpec((8, 8), ops.DType.F32)
    graph = ops.trace(
        sequential_weights,
        ops.Signature(
            (
                ops.Argument(spec, "x"),
                ops.Argument(spec, "first", ops.ValueKind.CONSTANT),
                ops.Argument(spec, "second", ops.ValueKind.CONSTANT),
            )
        ),
    )
    references = []

    def load(value):
        if references:
            assert references[-1]() is None
        array = np.eye(8, dtype=np.float32) * 2
        references.append(weakref.ref(array))
        return array

    result = evaluate_reference(
        graph, {graph.inputs[0]: np.ones((8, 8), np.float32)}, load=load, retain=set()
    )
    np.testing.assert_array_equal(result.outputs[0], np.full((8, 8), 4, np.float32))
    assert set(result.values) == set(graph.outputs)
    assert all(reference() is None for reference in references)
