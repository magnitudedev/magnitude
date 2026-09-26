import numpy as np
import pytest

import ops
from ops.tensor.primitive import evaluate_reference, round_reference


@pytest.mark.parametrize("dtype", [ops.DType.F16, ops.DType.BF16, ops.DType.F32])
def test_expert_reference_publishes_each_projection_before_route_weighting(dtype):
    spec = ops.TensorSpec((1, 2), dtype)
    routes = ops.TensorSpec((1, 2), ops.DType.I32)
    scores = ops.TensorSpec((1, 2), ops.DType.F32)
    weight = ops.TensorSpec((2, 2, 2), dtype)
    graph = ops.trace(ops.routed_experts, ops.Signature(tuple(ops.Argument(item)
                      for item in (spec, routes, scores, weight, weight, weight))))
    hidden = round_reference(np.array([[0.7, -0.3]], np.float32), dtype)
    gate = round_reference(np.array([[[1.3, 0.7], [0.3, -0.5]], [[0.9, -0.2], [0.2, 0.6]]], np.float32), dtype)
    up = round_reference(gate * np.float32(0.7), dtype)
    down = round_reference(gate * np.float32(1.1), dtype)
    route_values = np.array([[0, 1]], np.int32)
    coefficients = np.array([[0.03, 0.97]], np.float32)
    reference = evaluate_reference(graph, dict(zip(graph.inputs,
        (hidden, route_values, coefficients, gate, up, down), strict=True)))
    total = np.zeros((2,), np.float32)
    for rank in range(2):
        g = round_reference(gate[rank].astype(np.float32) @ hidden[0].astype(np.float32), dtype).astype(np.float32)
        u = round_reference(up[rank].astype(np.float32) @ hidden[0].astype(np.float32), dtype).astype(np.float32)
        activation = round_reference((g / (1 + np.exp(-g))) * u, dtype).astype(np.float32)
        projected = round_reference(down[rank].astype(np.float32) @ activation, dtype).astype(np.float32)
        total += coefficients[0, rank] * projected
    np.testing.assert_array_equal(reference.values[graph.outputs[0]], round_reference(total[None], dtype))
