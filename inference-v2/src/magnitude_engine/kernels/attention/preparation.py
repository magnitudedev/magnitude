"""Packed projection preparation composed from normalization and rotary finalization."""

import mlx.core as mx

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import Dispatch
from ..core.plan import Launch, Scalar, Source

PREPARATION = Source("attention/preparation.metal")


@kernel(source=PREPARATION)
def prepare_attention(
    projected,
    query_weight,
    key_weight,
    positions,
    frequencies,
    *,
    query_heads,
    kv_heads,
    width,
    query_eps,
    key_eps,
):
    arguments = {
        "projected": projected,
        "query_weight": query_weight,
        "key_weight": key_weight,
        "positions": positions,
        "frequencies": frequencies,
    }
    inputs = (
        projected.value.tensor,
        query_weight.value.tensor,
        key_weight.value.tensor,
        positions.value.tensor,
        frequencies.value.tensor,
    )
    projected, qw, kw, positions, frequencies = inputs
    batch, count = projected.shape[:2]
    hq, hk, width = (query_heads, kv_heads, width)
    if (
        projected.shape[-1] != (2 * hq + 2 * hk) * width
        or qw.shape != (width,)
        or kw.shape != (width,)
        or (frequencies.size * 2 > width)
        or (positions.size != batch)
    ):
        raise ValueError("packed attention projection disagrees with its head geometry")
    inferred = tuple(
        Tensor(shape, projected.dtype)
        for shape in (
            (batch, hq, count, width),
            (batch, hk, count, width),
            (batch, hk, count, width),
            (batch, count, hq * width),
        )
    )
    batch, count = inputs[0].shape[:2]
    threads = max(32, (width + 127) // 128 * 32)
    return Dispatch(
        arguments,
        dict(zip(("queries", "keys", "values", "gates"), inferred, strict=True)),
        Launch((threads, query_heads + kv_heads, batch * count), (threads, 1, 1)),
        (
            ("T", inputs[0].dtype),
            ("COUNT", count),
            ("HQ", query_heads),
            ("HK", kv_heads),
            ("WIDTH", width),
            ("THREADS", threads),
            ("ROTARY", inputs[4].size * 2),
        ),
        (Scalar("QEPS", query_eps), Scalar("KEPS", key_eps)),
    )


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
    batch = projected.shape[0]
    offsets = (
        mx.array([positions], mx.int32) if isinstance(positions, int) else positions.reshape(-1)
    )
    offsets = mx.broadcast_to(offsets, (batch,))
    return prepare_attention(
        projected,
        query_weight,
        key_weight,
        offsets,
        frequencies,
        query_heads=query_heads,
        kv_heads=kv_heads,
        width=width,
        query_eps=query_eps,
        key_eps=key_eps,
    )
