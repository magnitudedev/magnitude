"""Independent spatial attention checks, including non-square coordinates."""

import math

import numpy as np

import ops
from engine.models.qwen35.preparation import spatial_controls
from engine.models.qwen35.vision import (
    Affine,
    Normalization,
    VisionBlock,
    spatial_rotary,
    vision_block,
)


def test_spatial_rotary_pairs_each_axis_without_interchanging_heads():
    rng = np.random.default_rng(7)
    q, k = (rng.normal(size=(6, 3, 12)).astype(np.float32) for _ in range(2))
    positions = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [7, 19], [19, 7]], np.int32)
    signature = ops.Signature(
        tuple(
            ops.Argument(ops.TensorSpec(x.shape, dtype), name)
            for x, dtype, name in (
                (q, ops.DType.F32, "q"),
                (k, ops.DType.F32, "k"),
                (positions, ops.DType.I32, "coordinates"),
            )
        )
    )
    graph = ops.trace(spatial_rotary, signature)
    actual = ops.evaluate_reference(graph, {"q": q, "k": k, "coordinates": positions}).outputs
    frequencies = 10_000 ** (-np.arange(3, dtype=np.float64) / 3)
    angles = (positions[:, :, None] * frequencies).reshape(6, 1, 6)
    for value, output in zip((q, k), actual, strict=True):
        first, second = value[..., :6], value[..., 6:]
        expected = np.concatenate(
            (
                first * np.cos(angles) - second * np.sin(angles),
                second * np.cos(angles) + first * np.sin(angles),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)


def test_spatial_controls_follow_merge_order_and_interpolate_a_plane():
    coordinates, indices, coefficients = spatial_controls((1, 4, 6), 2, 5)
    assert coordinates[:8].tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ]
    table = np.array([2 * y + 3 * x for y in range(5) for x in range(5)])
    result = sum(
        table[index] * weight[:, 0] for index, weight in zip(indices, coefficients, strict=True)
    )
    expected = 2 * coordinates[:, 0] * 4 / 3 + 3 * coordinates[:, 1] * 4 / 5
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sum(coefficients), 1.0, atol=1e-7)


def test_vision_block_uses_full_attention_and_distinct_mlp_activation():
    rng = np.random.default_rng(19)
    rows, heads, width, intermediate = 4, 2, 8, 20
    hidden = heads * width
    data = {"x": rng.normal(size=(rows, hidden)).astype(np.float32)}
    data["coordinates"] = np.zeros((rows, 2), np.int32)
    data["visible"] = np.tile(np.array([[0, rows]], np.int32), (rows, 1))
    for index in range(3):
        data[f"selector{index}"] = np.array([index], np.int32)
    for name, shape in {
        "n1w": (hidden,),
        "n1b": (hidden,),
        "qw": (3 * hidden, hidden),
        "qb": (3 * hidden,),
        "pw": (hidden, hidden),
        "pb": (hidden,),
        "n2w": (hidden,),
        "n2b": (hidden,),
        "uw": (intermediate, hidden),
        "ub": (intermediate,),
        "dw": (hidden, intermediate),
        "db": (hidden,),
    }.items():
        data[name] = rng.normal(0, 0.2, shape).astype(np.float32)
    signature = ops.Signature(
        (),
        {
            name: ops.Argument(
                ops.TensorSpec(
                    value.shape, ops.DType.I32 if value.dtype == np.int32 else ops.DType.F32
                ),
                name,
            )
            for name, value in data.items()
        },
    )

    def equation(**x):
        return vision_block(
            x["x"],
            x["coordinates"],
            tuple(x[f"selector{i}"] for i in range(3)),
            x["visible"],
            VisionBlock(
                Normalization(x["n1w"], x["n1b"]),
                Affine(x["qw"], x["qb"]),
                Affine(x["pw"], x["pb"]),
                Normalization(x["n2w"], x["n2b"]),
                Affine(x["uw"], x["ub"]),
                Affine(x["dw"], x["db"]),
                heads,
            ),
        )

    def normalize(x, prefix):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6) * data[
            prefix + "w"
        ] + data[prefix + "b"]

    x = data["x"].astype(np.float64)
    qkv = (normalize(x, "n1") @ data["qw"].T + data["qb"]).reshape(rows, 3, heads, width)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    scores = np.einsum("ihd,jhd->hij", q, k) / math.sqrt(width)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    attended = np.einsum("hij,jhd->ihd", scores, v).reshape(rows, hidden)
    x += attended @ data["pw"].T + data["pb"]
    up = normalize(x, "n2") @ data["uw"].T + data["ub"]
    activated = 0.5 * up * (1 + np.tanh(math.sqrt(2 / math.pi) * (up + 0.044715 * up**3)))
    expected = x + activated @ data["dw"].T + data["db"]
    (actual,) = ops.evaluate_reference(ops.trace(equation, signature), data).outputs
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
