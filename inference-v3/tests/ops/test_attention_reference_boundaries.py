import numpy as np
import pytest

import ops


def trace(dtype=ops.DType.I32):
    return ops.trace(
        lambda query, history, ranges: ops.causal_attention(query, history, ranges),
        ops.Signature((
            ops.Argument(ops.TensorSpec((3, 4, 8), ops.DType.F32), "query"),
            ops.Argument(ops.TensorSpec((2, 7, 2, 8), ops.DType.F32), "history", ops.ValueKind.RESOURCE),
            ops.Argument(ops.TensorSpec((3, 2), dtype), "ranges"),
        )),
    )


def test_reference_reuses_head_storage_without_changing_vector_math():
    rng = np.random.default_rng(149)
    query = rng.normal(size=(3, 4, 8)).astype(np.float32)
    history = rng.normal(size=(2, 7, 2, 8)).astype(np.float32)
    ranges = np.array([[0, 0], [1, 4], [2, 5]], dtype=np.int32)
    expected = np.zeros_like(query)
    for row, (start, count) in enumerate(ranges):
        if count:
            for head in range(4):
                scores = query[row, head] @ history[0, start:start + count, head // 2].astype(np.float32).T
                scores *= 8**-0.5
                probabilities = np.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                expected[row, head] = probabilities @ history[1, start:start + count, head // 2].astype(np.float32)
    actual, = ops.evaluate_reference(trace(), {"query": query, "history": history, "ranges": ranges}).outputs
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("invalid", [(-1, 2), (0, -1), (5, 3)])
def test_reference_rejects_ranges_outside_storage(invalid):
    with pytest.raises(ValueError, match="outside cache"):
        ops.evaluate_reference(trace(), {
            "query": np.zeros((3, 4, 8), dtype=np.float32),
            "history": np.zeros((2, 7, 2, 8), dtype=np.float32),
            "ranges": np.array([invalid] * 3, dtype=np.int32),
        })


def test_visibility_is_an_integer_contract():
    with pytest.raises(ValueError, match="integer"):
        trace(ops.DType.F32)
