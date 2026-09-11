"""What ``Rounding.NATIVE_BF16`` means, as TIR.

The reference framework rounds to BF16 after every operation. These macros are
that definition: each boundary rounds rather than carrying one FP32 intermediate
across a whole expression. They are reached as ``kernels.precision.native_*``;
the value object itself stays importable without a compiler.
"""

import tilelang.language as T


@T.macro
def native_sigmoid(x, precise=True):
    exponential = (
        (T.exp(T.abs(x)) if precise else T.__exp(T.abs(x))).astype("bfloat16").astype("float32")
    )
    denominator = (1 + exponential).astype("bfloat16").astype("float32")
    reciprocal = (1 / denominator).astype("bfloat16").astype("float32")
    return T.if_then_else(x < 0, reciprocal, 1 - reciprocal).astype("bfloat16")


@T.macro
def native_decay(input, bias, log_rate):
    x = (input + bias).astype("bfloat16").astype("float32")
    exponential = T.exp(-T.abs(x)).astype("bfloat16").astype("float32")
    logged = (
        T.if_then_else(exponential < 1e-4, exponential, T.log(1 + exponential))
        .astype("bfloat16")
        .astype("float32")
    )
    softplus = (T.max(x, 0) + logged).astype("bfloat16").astype("float32")
    return T.exp(-T.exp(log_rate) * softplus)
