"""Platform-independent arithmetic for the transforms that compute values rather than copy them.

`torch.exp` differs by a unit in the last place between CPU architectures (libm and vector kernels), so a conversion
run on two hosts would write different bytes and the slice's identity would depend on the host. Each function here is
the correctly rounded result, from integer and decimal arithmetic only, so every host writes the same bytes."""
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
import numpy as np

_F32_MANTISSA = 24
_F32_MIN_EXPONENT, _F32_MAX_EXPONENT = -126, 127  # of the leading bit of a normal F32


def _round_f32(value: Decimal) -> float:
    """The F32 nearest a positive finite `value` (ties to even), as the exact float it is; refused outside the normal range."""
    with localcontext() as ctx:
        ctx.prec = 80
        # a first guess, corrected below; Decimal's own logarithm, since a value past F32 is past float's range too
        exponent = int((value.ln() / Decimal(2).ln()).to_integral_value(ROUND_FLOOR))
        while Decimal(2) ** exponent > value:
            exponent -= 1
        while Decimal(2) ** (exponent + 1) <= value:
            exponent += 1
        if not _F32_MIN_EXPONENT <= exponent <= _F32_MAX_EXPONENT:
            raise ValueError(f"{value} is outside F32's normal range")
        mantissa = int((value / Decimal(2) ** (exponent - (_F32_MANTISSA - 1))).to_integral_value(ROUND_HALF_EVEN))
        return float(mantissa) * 2.0 ** (exponent - (_F32_MANTISSA - 1))  # exact: a 24-bit integer times a power of two


def negative_exp_f32(values: np.ndarray) -> np.ndarray:
    """`-exp(x)` of each F32 `x`, correctly rounded to F32."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    with localcontext() as ctx:
        ctx.prec = 80
        out = np.fromiter((-_round_f32(Decimal(float(x)).exp()) for x in flat), dtype=np.float64, count=flat.size)
    return out.astype(np.float32).reshape(values.shape)
