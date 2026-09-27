"""Qwen's stateless image encoder expressed entirely through Ops."""

from dataclasses import dataclass

import ops


@dataclass(frozen=True)
class Affine:
    weight: ops.Tensor
    bias: ops.Tensor

    def __call__(self, value):
        return ops.linear(value, self.weight, self.bias)


@dataclass(frozen=True)
class Normalization:
    weight: ops.Tensor
    bias: ops.Tensor

    def __call__(self, value):
        return ops.layer_norm(value, self.weight, self.bias, epsilon=1e-6)


@dataclass(frozen=True)
class VisionBlock:
    norm1: Normalization
    qkv: Affine
    projection: Affine
    norm2: Normalization
    up: Affine
    down: Affine
    heads: int


@ops.formula(id="qwen35.vision_rotary", version=1, metric="tokens", rows="query")
def spatial_rotary(query, key, coordinates):
    """Rotate height and width quarter-pairs with their own frequency ladder.

    An axis becomes a leading row, so the ordinary rotary contract expresses
    the operation without backend-specific math or a second rotary primitive.
    """
    rows, heads, width = query.shape
    if width % 4 or coordinates.shape != (rows, 2):
        raise ValueError("vision rotary requires paired spatial axes and quarter channels")

    def separate(value):
        value = ops.reshape(value, (rows, heads, 2, 2, width // 4))
        value = ops.transpose(value, (0, 3, 1, 2, 4))
        return ops.reshape(value, (rows * 2, heads, width // 2))

    query, key = ops.rotary(
        separate(query), separate(key), ops.reshape(coordinates, (rows * 2,)), base=10_000.0
    )

    def restore(value):
        value = ops.reshape(value, (rows, 2, heads, 2, width // 4))
        return ops.reshape(ops.transpose(value, (0, 2, 3, 1, 4)), (rows, heads, width))

    return restore(query), restore(key)


@ops.formula(id="qwen35.vision_block", version=1, metric="tokens", rows="hidden")
def vision_block(hidden, coordinates, selectors, visible, weights: VisionBlock):
    rows, channels = hidden.shape
    width = channels // weights.heads
    projected = weights.qkv(weights.norm1(hidden))
    projected = ops.transpose(ops.reshape(projected, (rows, 3, weights.heads, width)), (1, 0, 2, 3))
    query, key, value = (
        ops.reshape(ops.take_rows(projected, selector), (rows, weights.heads, width))
        for selector in selectors
    )
    query, key = spatial_rotary(query, key, coordinates)
    history = ops.concatenate(
        tuple(ops.reshape(item, (1, rows, weights.heads, width)) for item in (key, value))
    )
    # One image is one complete bidirectional attention domain.
    attended = ops.causal_attention(query, history, visible, sequence_count=1)
    hidden = hidden + weights.projection(ops.reshape(attended, (rows, channels)))
    return hidden + weights.down(ops.gelu_tanh(weights.up(weights.norm2(hidden))))


@ops.formula(id="qwen35.vision_merger", version=1, metric="tokens", rows="hidden")
def merge_patches(hidden, norm: Normalization, up: Affine, down: Affine, merge: int):
    rows, channels = hidden.shape
    normalized = ops.reshape(norm(hidden), (rows // merge**2, channels * merge**2))
    return down(ops.gelu(up(normalized)))
