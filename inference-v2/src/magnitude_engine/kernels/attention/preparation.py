"""Packed projection preparation composed from normalization and rotary finalization."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.graph import Tensor, signature
from ..core.kernel import Kernel
from ..core.plan import Launch, Scalar, Source
from ..core.primitive import Primitive

PREPARATION = Source("attention/preparation.metal")


@dataclass(frozen=True)
class Preparation(Primitive):
    """Unpack Q/gate/K/V, native RMS normalization, then FP32 rotary pairs."""

    query_heads: int
    kv_heads: int
    width: int
    query_eps: float
    key_eps: float

    def infer(self, inputs):
        projected, qw, kw, positions, frequencies = inputs
        batch, count = projected.shape[:2]
        hq, hk, width = self.query_heads, self.kv_heads, self.width
        if (
            projected.shape[-1] != (2 * hq + 2 * hk) * width
            or qw.shape != (width,)
            or kw.shape != (width,)
            or frequencies.size * 2 > width
            or positions.size != batch
        ):
            raise ValueError("packed attention projection disagrees with its head geometry")
        return tuple(
            Tensor(shape, projected.dtype)
            for shape in (
                (batch, hq, count, width),
                (batch, hk, count, width),
                (batch, hk, count, width),
                (batch, count, hq * width),
            )
        )

    def lower(self, inputs):
        batch, count = inputs[0].shape[:2]
        threads = max(32, ((self.width + 127) // 128) * 32)
        return Kernel(
            signature(
                ("projected", "query_weight", "key_weight", "positions", "frequencies"), inputs
            ),
            signature(("queries", "keys", "values", "gates"), self.infer(inputs)),
            PREPARATION,
            Launch((threads, self.query_heads + self.kv_heads, batch * count), (threads, 1, 1)),
            (
                ("T", inputs[0].dtype),
                ("COUNT", count),
                ("HQ", self.query_heads),
                ("HK", self.kv_heads),
                ("WIDTH", self.width),
                ("THREADS", threads),
                ("ROTARY", inputs[4].size * 2),
            ),
            (Scalar("QEPS", self.query_eps), Scalar("KEPS", self.key_eps)),
        ).bind()


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
    return Preparation(query_heads, kv_heads, width, query_eps, key_eps)(
        projected, query_weight, key_weight, offsets, frequencies
    )
