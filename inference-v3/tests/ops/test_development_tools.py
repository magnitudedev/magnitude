
from tests.ops.target_fixture import matrix_query
import pytest

import ops
import numpy as np
from ops.lab import Fixture

CAPABILITIES = ops.CompilerTarget(
    32,
    256,
    32 * 1024,
    matrix_query=matrix_query((
        ops.MatrixTile(8, 8, 8, ops.DType.F16, ops.DType.F32),
        ops.MatrixTile(8, 8, 8, ops.DType.F32, ops.DType.F32),
    )),


    identity="development-tool-test",
)


def _encoded(shape):
    return ops.TensorSpec(shape, ops.DType.F16).with_representation(
        ops.Affine(
            ops.Code(4),
            64,
            ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16),
        )
    )


def test_small_parallel_projections_form_one_lowering_region():
    source = ops.TensorSpec((1, 512), ops.DType.F16)
    weight = _encoded((32, 512))

    @ops.formula(id="test.parallel-projections")
    def project(value, first, second, third):
        return (
            ops.linear(value, first),
            ops.linear(value, second),
            ops.linear(value, third),
        )

    ops.operation(project)(ops.operation_bodies.parallel_projections)
    plan = ops.analyze(
        project,
        signature=ops.Signature(
            (
                ops.Argument(source, "source"),
                ops.Argument(weight, "first", ops.ValueKind.CONSTANT),
                ops.Argument(weight, "second", ops.ValueKind.CONSTANT),
                ops.Argument(weight, "third", ops.ValueKind.CONSTANT),
            )
        ),
        compiler_target=CAPABILITIES,
        options=ops.CompileOptions(mode="decode"),
        available_bytes=1 << 30,
    )

    assert [candidate.name.split("@", 1)[0] for candidate in plan.operations] == [
        "linear.parallel-packet-decode"
    ]
    assert plan.diagnostics.dispatches == 1
    assert len(plan.submissions) == 1


def test_prepared_fixture_derives_only_requested_boundary_and_reuses_it():
    @ops.formula(id="test.fixture.sequence")
    def sequence(value):
        return ops.silu(ops.tanh(value))

    graph = ops.trace(sequence, ops.Signature((ops.Argument(ops.TensorSpec((4,), ops.DType.F32)),)))
    original = np.array([-1, 0, 1, 2], dtype=np.float32)
    fixture = Fixture.from_inputs(graph, {graph.inputs[0]: original})
    target, = ops.FormulaTree(graph).occurrences(ops.silu)
    boundary = fixture.boundary(target)
    original[:] = 99
    np.testing.assert_array_equal(next(iter(boundary.inputs.values())).reference,
                                  np.tanh(np.array([-1, 0, 1, 2], dtype=np.float32)))
    assert fixture.boundary(target) is boundary
    assert boundary.reference is boundary.reference


def test_prepared_fixture_rejects_unknown_root_input():
    graph = ops.trace(ops.tanh, ops.Signature((ops.Argument(ops.TensorSpec((4,), ops.DType.F32)),)))
    with pytest.raises(ValueError, match="declared production inputs"):
        Fixture.from_inputs(graph, {graph.outputs[0]: np.zeros(4, dtype=np.float32)})
