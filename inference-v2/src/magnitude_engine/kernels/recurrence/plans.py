"""Ordered delta transition with endpoint-only and output-producing schedules."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.graph import Tensor, signature
from ..core.kernel import Kernel
from ..core.plan import Launch, Scalar, Source
from ..core.primitive import Primitive

DELTA = Source("recurrence/delta.metal")


@dataclass(frozen=True)
class Delta(Primitive):
    """Sequential FP32 decay, prediction, correction and query projection.

    A lane owns DK/32 consecutive state coordinates. Each prediction/query uses
    that fixed 32-lane reduction; the complete transition precedes the next token.
    State ownership is independent across batch, head and value coordinate.
    """

    state_only: bool = False
    specialize_prefill: bool = False

    def infer(self, inputs):
        q, k, v, decay, beta, state, length = inputs
        batch, tokens, hk, dk = k.shape
        hv, dv = v.shape[2:]
        if (
            q != k
            or dk % 32
            or hv % hk
            or v.shape[:2] != (batch, tokens)
            or state != Tensor((batch, hv, dv, dk), mx.float32)
            or decay.shape != (batch, tokens, hv)
            or beta.shape != decay.shape
        ):
            raise ValueError("delta operands disagree with the ordered transition geometry")
        return (state,) if self.state_only else (state, Tensor(v.shape, q.dtype))

    def lower(self, inputs):
        batch, tokens, hk, dk = inputs[1].shape
        hv, dv = inputs[2].shape[2:]
        return Kernel(
            signature(("q", "k", "v", "decay", "beta", "initial", "length"), inputs),
            signature(("final",) if self.state_only else ("final", "output"), self.infer(inputs)),
            DELTA,
            Launch((32, dv, batch * hv), (32, 4, 1)),
            (
                ("In", inputs[0].dtype),
                ("TOKENS", tokens),
                ("HK", hk),
                ("DK", dk),
                ("HV", hv),
                ("DV", dv),
                ("FIXED_TOKENS", tokens <= 8 or self.specialize_prefill),
            ),
            (Scalar("STATE_ONLY", self.state_only),),
        ).bind()


def advance(q, k, v, decay, beta, state, *, state_only=False, specialize_prefill=False):
    return Delta(state_only, specialize_prefill)(
        q, k, v, decay, beta, state, mx.array([k.shape[1]], mx.int32)
    )
