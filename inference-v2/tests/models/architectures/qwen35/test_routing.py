import mlx.core as mx
import pytest

from magnitude_engine.kernels.reductions.routing import select


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("experts", [32, 137, 256, 512, 1024])
@pytest.mark.parametrize("normalize", [False, True])
def test_routing_preserves_upstream_probabilities_indices_and_shared_gate(
    dtype, experts, normalize
):
    mx.random.seed(521)
    # Include exact and rounding-induced ties, and saturated shared gates.
    logits = mx.random.normal((1, 4, experts + 1)).astype(dtype)
    logits[0, 1, :] = 0
    logits[0, 2, :] = mx.floor(logits[0, 2, :] * 2) / 2
    logits[0, 3, -1] = -80
    actual_indices, actual_scores, actual_shared = select(logits, 8, normalize)
    probabilities = mx.softmax(logits[..., :-1], precise=True, axis=-1)
    indices = mx.argpartition(probabilities, kth=-8, axis=-1)[..., -8:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    if normalize:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    assert mx.array_equal(actual_indices, indices).item()
    assert mx.allclose(actual_scores, scores, atol=2e-7, rtol=2e-6).item()
    assert mx.array_equal(actual_shared, mx.sigmoid(logits[..., -1:])).item()
