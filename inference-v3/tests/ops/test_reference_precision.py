"""Declared publication precision is shared by all formula reference paths."""

import numpy as np

import ops
from ops.tensor.primitive import round_reference


def test_bfloat_reference_rounds_ties_to_even_and_preserves_nan():
    bits = np.array([0x3F808000, 0x3F818000, 0x80000000, 0x7F800001], dtype=np.uint32)
    result = round_reference(bits.view(np.float32), ops.DType.BF16).view(np.uint32)
    np.testing.assert_array_equal(result[:3], [0x3F800000, 0x3F820000, 0x80000000])
    assert np.isnan(result[3:].view(np.float32)[0])


def test_bfloat_cast_is_an_actual_reference_rounding_boundary():
    spec = ops.TensorSpec((2,), ops.DType.F32)
    graph = ops.trace(lambda value: ops.cast(value, ops.DType.BF16),
                      ops.Signature((ops.Argument(spec, "input"),)))
    values = np.array([1.00390625, 1.01171875], dtype=np.float32)
    actual, = ops.evaluate_reference(graph, {"input": values}).outputs
    np.testing.assert_array_equal(actual, np.array([1.0, 1.015625], dtype=np.float32))
    np.testing.assert_array_equal(values, [1.00390625, 1.01171875])
