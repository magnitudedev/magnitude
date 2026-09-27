"""Ordered delta transition with endpoint-only and output-producing schedules."""

import mlx.core as mx

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import Dispatch
from ..core.plan import Launch, Scalar, Source

DELTA = Source("recurrence/delta.metal")


@kernel(source=DELTA)
def delta(q, k, v, decay, beta, initial, length, *, state_only=False, specialize_prefill=False):
    arguments = {
        "q": q,
        "k": k,
        "v": v,
        "decay": decay,
        "beta": beta,
        "initial": initial,
        "length": length,
    }
    inputs = (
        q.value.tensor,
        k.value.tensor,
        v.value.tensor,
        decay.value.tensor,
        beta.value.tensor,
        initial.value.tensor,
        length.value.tensor,
    )
    q, k, v, decay, beta, state, length = inputs
    batch, tokens, hk, dk = k.shape
    hv, dv = v.shape[2:]
    if (
        q != k
        or dk % 32
        or hv % hk
        or (v.shape[:2] != (batch, tokens))
        or (state != Tensor((batch, hv, dv, dk), mx.float32))
        or (decay.shape != (batch, tokens, hv))
        or (beta.shape != decay.shape)
    ):
        raise ValueError("delta operands disagree with the ordered transition geometry")
    inferred = (state,) if state_only else (state, Tensor(v.shape, q.dtype))
    batch, tokens, hk, dk = inputs[1].shape
    hv, dv = inputs[2].shape[2:]
    return Dispatch(
        arguments,
        dict(zip(("final",) if state_only else ("final", "output"), inferred, strict=True)),
        Launch((32, dv, batch * hv), (32, 4, 1)),
        (
            ("In", inputs[0].dtype),
            ("TOKENS", tokens),
            ("HK", hk),
            ("DK", dk),
            ("HV", hv),
            ("DV", dv),
            ("FIXED_TOKENS", tokens <= 8 or specialize_prefill),
        ),
        (Scalar("STATE_ONLY", state_only),),
    )


def advance(q, k, v, decay, beta, state, *, state_only=False, specialize_prefill=False):
    result = delta(
        q,
        k,
        v,
        decay,
        beta,
        state,
        mx.array([k.shape[1]], mx.int32),
        state_only=state_only,
        specialize_prefill=specialize_prefill,
    )
    return (result,) if state_only else result
