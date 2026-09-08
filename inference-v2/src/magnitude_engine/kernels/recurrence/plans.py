"""Ordered delta transition with endpoint-only and output-producing plans."""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Scalar, Source

DELTA = Program("magnitude_delta", Source("recurrence/delta.metal"))


def advance(q, k, v, decay, beta, state, *, state_only=False, specialize_prefill=False):
    batch, tokens, hk, dk = k.shape
    hv, dv = v.shape[2:]
    return KernelPlan(
        program=DELTA,
        inputs=(
            Input("q", q),
            Input("k", k),
            Input("v", v),
            Input("decay", decay),
            Input("beta", beta),
            Input("initial", state),
            Input("length", mx.array([tokens], mx.int32)),
        ),
        outputs=(Output("final", state.shape, mx.float32),)
        if state_only
        else (Output("final", state.shape, mx.float32), Output("output", v.shape, q.dtype)),
        launch=Launch((32, dv, batch * hv), (32, 4, 1)),
        template=(
            ("In", q.dtype),
            ("DK", dk),
            ("DV", dv),
            ("HK", hk),
            ("HV", hv),
            ("SHORT_T", tokens if tokens <= 8 or specialize_prefill else 0),
        ),
        constants=(Scalar("STATE_ONLY", state_only),),
    ).run()
