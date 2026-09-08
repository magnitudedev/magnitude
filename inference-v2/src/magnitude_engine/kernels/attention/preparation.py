"""Unpack, normalize and rotate a packed text-attention projection in one pass."""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Scalar, Source

PREPARATION = Program("magnitude_qwen_attention_prepare", Source("attention/preparation.metal"))


def prepare(
    projected,
    query_weight,
    key_weight,
    positions,
    frequencies,
    *,
    query_heads: int,
    kv_heads: int,
    width: int,
    query_eps: float,
    key_eps: float,
):
    batch, count = projected.shape[:2]
    offsets = (
        mx.array([positions], mx.int32) if isinstance(positions, int) else positions.reshape(-1)
    )
    offsets = mx.broadcast_to(offsets, (batch,))
    threads = max(32, ((width + 127) // 128) * 32)
    return tuple(
        KernelPlan(
            program=PREPARATION,
            inputs=(
                Input("projected", projected),
                Input("query_weight", query_weight),
                Input("key_weight", key_weight),
                Input("positions", offsets),
                Input("frequencies", frequencies),
            ),
            outputs=(
                Output("queries", (batch, query_heads, count, width), projected.dtype),
                Output("keys", (batch, kv_heads, count, width), projected.dtype),
                Output("values", (batch, kv_heads, count, width), projected.dtype),
                Output("gates", (batch, count, query_heads * width), projected.dtype),
            ),
            launch=Launch((threads, query_heads + kv_heads, batch * count), (threads, 1, 1)),
            template=(
                ("T", projected.dtype),
                ("HQ", query_heads),
                ("HK", kv_heads),
                ("WIDTH", width),
                ("COUNT", count),
                ("ROTARY", frequencies.size * 2),
                ("THREADS", threads),
            ),
            constants=(
                Scalar("QEPS", query_eps),
                Scalar("KEPS", key_eps),
            ),
        ).run()
    )
