import ops
from ops.lab.ceilings import output_storage_floor


def test_output_floor_counts_shared_backing_once_and_excludes_borrowed_views():
    spec = ops.TensorSpec((4,), ops.DType.F32)

    def function(value):
        result = value + value
        return result, ops.reshape(result, (2, 2)), ops.reshape(value, (2, 2))

    graph = ops.trace(function, ops.Signature((ops.Argument(spec),)))
    assert output_storage_floor(graph) == 16
    assert graph.alias_root(graph.outputs[0]) == graph.alias_root(graph.outputs[1])
    assert graph.alias_root(graph.outputs[2]) == graph.inputs[0]


def test_view_only_formula_has_no_positive_fresh_output_storage_bound():
    graph = ops.trace(lambda value: ops.reshape(value, (2, 2)),
                      ops.Signature((ops.Argument(ops.TensorSpec((4,), ops.DType.F32)),)))
    assert output_storage_floor(graph) == 0


def test_reshape_retains_mutable_resource_identity():
    spec = ops.TensorSpec((4,), ops.DType.U8)
    graph = ops.trace(lambda value: ops.reshape(value, (2, 2)),
                      ops.Signature((ops.Argument(spec, "state", ops.ValueKind.RESOURCE),)))
    original = graph.value(graph.resources[0])
    result = graph.value(graph.outputs[0])
    assert result.resource_id == original.resource_id
    assert result.resource_version == original.resource_version
    assert output_storage_floor(graph) == 0
