"""RMS reduction plans with explicit native rounding and fused finalization."""

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Scalar, Source

RESIDUAL_NORM = Program("magnitude_residual_norm", Source("reductions/residual_norm.metal"))

GATED_NORM = Program("magnitude_gated_norm", Source("reductions/gated_norm.metal"))


def residual_norm(x, update, weight, eps):
    width = x.shape[-1]
    threads = min(width // 4, 1024)
    result = KernelPlan(
        program=RESIDUAL_NORM,
        inputs=(Input("x", x), Input("a", update), Input("w1", weight)),
        outputs=(Output("xnew", x.shape, x.dtype), Output("normalized", x.shape, x.dtype)),
        launch=Launch((threads, x.size // width, 1), (threads, 1, 1)),
        template=(
            ("T", x.dtype),
            ("D", width),
            ("THREADS", threads),
            ("NCHUNK", (width + threads * 4 - 1) // (threads * 4)),
        ),
        constants=(Scalar("EPS", eps),),
    ).run()
    return (result[0], result[1])


def gated_norm(hidden, gate, weight, eps):
    width = hidden.shape[-1]
    return KernelPlan(
        program=GATED_NORM,
        inputs=(Input("x", hidden), Input("gate", gate), Input("w", weight)),
        outputs=(Output("out", hidden.shape, hidden.dtype),),
        launch=Launch((width // 4, hidden.size // width, 1), (width // 4, 1, 1)),
        template=(("T", hidden.dtype), ("D", width)),
        constants=(Scalar("EPS", eps),),
    ).run()[0]
