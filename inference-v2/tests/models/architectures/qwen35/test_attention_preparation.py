import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

from magnitude_engine.models.architectures.qwen35.attention.preparation import prepare


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch,count,width,rotary", [(1, 1, 256, 64), (2, 4, 128, 128), (1, 8, 64, 32)]
)
def test_packed_attention_preparation_matches_upstream(dtype, batch, count, width, rotary):
    mx.random.seed(486)
    hq, hk = 8, 2
    projected = mx.random.normal((batch, count, (hq + hk) * 2 * width)).astype(dtype)
    qn, kn = nn.RMSNorm(width, eps=1e-6), nn.RMSNorm(width, eps=1e-5)
    qn.weight = mx.random.uniform(0.5, 1.5, (width,)).astype(dtype)
    kn.weight = mx.random.uniform(0.5, 1.5, (width,)).astype(dtype)
    rope = Qwen3_5RotaryEmbedding(rotary, base=1000000, mrope_section=[rotary // 2, 0, 0])
    positions = mx.array([65536 + 127 * b for b in range(batch)], mx.int32)
    actual = prepare(
        projected,
        qn.weight,
        kn.weight,
        positions,
        rope.inv_freq,
        query_heads=hq,
        kv_heads=hk,
        width=width,
        query_eps=qn.eps,
        key_eps=kn.eps,
    )
    qg, k, v = mx.split(projected, [2 * hq * width, (2 * hq + hk) * width], axis=-1)
    q, g = mx.split(qg.reshape(batch, count, hq, 2 * width), 2, axis=-1)
    q = qn(q).transpose(0, 2, 1, 3)
    k = kn(k.reshape(batch, count, hk, width)).transpose(0, 2, 1, 3)
    q, k = rope.apply_rotary(q, k, positions[:, None] + mx.arange(count)[None, :])
    expected = (
        q,
        k,
        v.reshape(batch, count, hk, width).transpose(0, 2, 1, 3),
        g.reshape(batch, count, -1),
    )
    for a, e in zip(actual, expected, strict=True):
        assert mx.allclose(a, e, atol=0.002, rtol=0.002).item()
