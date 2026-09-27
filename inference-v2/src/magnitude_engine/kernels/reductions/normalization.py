"""Handwritten RMS rows composed with ordinary MLX residual and gating arithmetic."""

import mlx.core as mx

from .. import compile, kernel
from ..core import metal


@kernel(source="rms.metal", function="magnitude_rms")
def rms_norm(x, weight, *, eps=1e-6):
    width = x.shape[-1]
    if width < 128 or width % 128:
        raise ValueError("RMS requires feature width divisible by 128")
    if weight.shape != (width,) or weight.dtype != x.dtype:
        raise ValueError("RMS weights must match feature width and dtype")
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("unsupported RMS dtype")
    return metal.Rows(
        x,
        threads=min(width // 4, 1024),
        parameters={"weight": weight, "eps": eps},
        scratch={"partial": metal.Scratch(mx.float32, 32)},
    )


@compile
def residual_norm(x, update, weight, eps):
    residual = (x + update).astype(x.dtype)
    return residual, rms_norm(residual, weight, eps=eps)


@compile
def gated_norm(hidden, gate, weight, eps):
    normalized = rms_norm(hidden, weight, eps=eps)
    z = gate.astype(mx.float32)
    return ((z * mx.sigmoid(z)) * normalized.astype(mx.float32)).astype(hidden.dtype)
