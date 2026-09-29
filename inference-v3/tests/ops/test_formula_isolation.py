"""Typed boundaries survive graph position, composition and implementation changes."""

import numpy as np
import pytest

import ops
from ops.tensor.graph import prune_dead_nodes


@ops.formula
def projection(hidden, weight):
    return ops.linear(hidden, weight)


@ops.formula
def activated(hidden, weight):
    return ops.silu(projection(hidden, weight))


@ops.formula
def pair(hidden, first, second):
    return activated(hidden, first), activated(hidden, second)


def signature():
    return ops.Signature((
        ops.Argument(ops.TensorSpec((2, 3), ops.DType.F32), "hidden"),
        ops.Argument(ops.TensorSpec((4, 3), ops.DType.F32), "first", ops.ValueKind.CONSTANT),
        ops.Argument(ops.TensorSpec((4, 3), ops.DType.F32), "second", ops.ValueKind.CONSTANT),
    ))


def test_typed_occurrence_isolates_actual_ports_and_independent_reference():
    graph = ops.trace(pair, signature())
    tree = ops.FormulaTree(graph)
    parent, = tree.roots
    first, second = parent.occurrences(activated)
    assert first != second
    assert first.parent == second.parent == parent
    isolated = second.isolate()
    assert tuple(node.operation for node in isolated.graph.nodes) == ("linear", "silu")
    assert isolated.graph.inputs == (0,)
    assert isolated.graph.constants == (1,)
    assert isolated.inputs[1].original == graph.constants[1]
    hidden = np.arange(6, dtype=np.float32).reshape(2, 3)
    weight = np.ones((4, 3), dtype=np.float32)
    bindings = isolated.bind({graph.inputs[0]: hidden, graph.constants[1]: weight})
    result = ops.evaluate_reference(isolated.graph, bindings).outputs[0]
    expected = hidden @ weight.T
    np.testing.assert_allclose(result, expected / (1 + np.exp(-expected)))
    child, = ops.FormulaTree(isolated.graph).roots
    assert child.semantic_identity == second.semantic_identity


def test_formula_history_identity_ignores_occurrence_and_graph_positions():
    first_graph = ops.trace(pair, signature(), name="first-enclosing-function")
    second_graph = ops.trace(lambda x, a, b: pair(ops.tanh(x), a, b), signature(), name="other")
    a, b = ops.FormulaTree(first_graph).occurrences(projection)
    c, d = ops.FormulaTree(second_graph).occurrences(projection)
    assert a.semantic_identity == b.semantic_identity == c.semantic_identity == d.semantic_identity
    assert a != c
    with pytest.raises(ValueError, match="belong to this trace"):
        ops.FormulaTree(second_graph).affected((a,))
    with pytest.raises(TypeError, match="Formula object"):
        ops.FormulaTree(first_graph).occurrences("projection")


def test_composed_change_invalidates_ancestors_without_unrelated_sibling():
    tree = ops.FormulaTree(ops.trace(pair, signature()))
    parent, = tree.roots
    first, second = parent.occurrences(activated)
    projection_handle, = first.occurrences(projection)
    activation_handle, = first.occurrences(ops.silu)
    assert set(tree.affected((projection_handle,))) == {projection_handle, activation_handle, first, parent}
    assert second not in tree.affected((projection_handle,))


def test_consumer_dependencies_follow_primitive_edges_between_formula_calls():
    @ops.formula
    def source(value):
        return ops.tanh(value)

    @ops.formula
    def consumer(value):
        return ops.silu(value)

    graph = ops.trace(lambda x: consumer(source(x) + x),
                      ops.Signature((ops.Argument(ops.TensorSpec((4,), ops.DType.F32)),)))
    tree = ops.FormulaTree(graph)
    producer, = tree.occurrences(source)
    dependent, = tree.occurrences(consumer)
    activation, = dependent.occurrences(ops.silu)
    assert set(tree.affected((producer,))) == {producer, dependent, activation}


def test_partial_pruned_call_cannot_claim_complete_formula_measurement():
    raw = ops.trace(lambda x, a, b: pair(x, a, b)[0], signature())
    pruned = prune_dead_nodes(raw)
    parent, = ops.FormulaTree(pruned).roots
    with pytest.raises(ValueError, match="original formula trace"):
        parent.isolate()
    with pytest.raises(ValueError, match="partial occurrence"):
        _ = parent.semantic_identity
    complete, = ops.FormulaTree(raw).roots
    assert len(complete.isolate().graph.outputs) == 2


def test_aliased_ports_are_bound_once_but_retain_both_paths():
    @ops.formula
    def add(left, right):
        return left + right

    graph = ops.trace(lambda x: add(x, x),
                      ops.Signature((ops.Argument(ops.TensorSpec((4,), ops.DType.F32)),)))
    handle, = ops.FormulaTree(graph).roots
    isolated = handle.isolate()
    assert len(isolated.inputs) == 1
    assert isolated.inputs[0].paths == (("left",), ("right",))


def test_tensor_closure_is_rejected_instead_of_becoming_hidden_fixture_input():
    def outer(x):
        @ops.formula
        def bad(y):
            return y + x

        return bad(ops.tanh(x))

    with pytest.raises(ValueError, match="explicit arguments"):
        ops.trace(outer, ops.Signature((ops.Argument(ops.TensorSpec((4,), ops.DType.F32)),)))


def test_state_boundary_preserves_versioned_aliases_at_later_occurrence():
    @ops.formula
    def append(history, keys, values, destinations):
        return ops.kv_append(history, keys, values, destinations)

    def twice(history, keys, values, destinations):
        return append(append(history, keys, values, destinations), keys, values, destinations)

    graph = ops.trace(twice, ops.Signature((
        ops.Argument(ops.TensorSpec((2, 8, 2, 4), ops.DType.F32), "history", ops.ValueKind.RESOURCE),
        ops.Argument(ops.TensorSpec((2, 2, 4), ops.DType.F32), "keys"),
        ops.Argument(ops.TensorSpec((2, 2, 4), ops.DType.F32), "values"),
        ops.Argument(ops.TensorSpec((2,), ops.DType.I32), "destinations"),
    )))
    first, second = ops.FormulaTree(graph).occurrences(append)
    isolated = second.isolate()
    resource, = isolated.graph.resources
    assert isolated.graph.value(resource).producer is None
    assert isolated.graph.value(resource).resource_version == 1
    assert isolated.graph.nodes[0].effects.writes == ((0, 1, 2),)
    assert first.semantic_identity == second.semantic_identity
    assert ops.FormulaTree(isolated.graph).roots[0].semantic_identity == second.semantic_identity


def test_caught_formula_failure_does_not_leave_orphan_trace_nodes_or_occurrences():
    @ops.formula
    def rejected(value):
        ops.tanh(value)
        raise ValueError("rejected shape")

    @ops.formula
    def accepted(value):
        try:
            rejected(value)
        except ValueError:
            pass
        return ops.silu(value)

    graph = ops.trace(accepted, ops.Signature((ops.Argument(ops.TensorSpec((4,), ops.DType.F32)),)))
    assert tuple(node.operation for node in graph.nodes) == ("silu",)
    assert len(graph.formulas.calls) == 2
    root, = ops.FormulaTree(graph).roots
    assert root.definition == accepted.ref
    assert tuple(child.definition for child in root.children) == (ops.silu.ref,)
    assert not ops.FormulaTree(graph).occurrences(rejected)
