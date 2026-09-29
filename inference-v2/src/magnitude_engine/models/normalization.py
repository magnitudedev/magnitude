"""Residual and gated normalization contracts, with qualified native kernel plans."""

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.kernels.reductions import normalization


def residual_norm(x: mx.array, update: mx.array, norm) -> tuple[mx.array, mx.array]:
    width = x.shape[-1]
    if not isinstance(norm, nn.RMSNorm) or width % 128 or x.dtype != update.dtype:
        residual = x + update
        return residual, norm(residual)
    return normalization.residual_norm(x, update, norm.weight, norm.eps)


class GatedRMSNorm(nn.Module):
    def __init__(self, weight: mx.array, eps: float):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, hidden: mx.array, gate: mx.array) -> mx.array:
        width = hidden.shape[-1]
        if width % 128 or width > 4096:
            normalized = mx.fast.rms_norm(hidden, self.weight, self.eps)
            return (nn.silu(gate.astype(mx.float32)) * normalized.astype(mx.float32)).astype(
                hidden.dtype
            )
        return normalization.gated_norm(hidden, gate, self.weight, self.eps)
