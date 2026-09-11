"""Shared checks and tiling policy for direct affine schedules."""

from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import Affine, DirectCoefficients, WeightLayout


def check(layout: WeightLayout) -> tuple[Affine, DirectCoefficients]:
    representation = layout.representation
    if not isinstance(representation, Affine) or not isinstance(
        representation.coefficients, DirectCoefficients
    ):
        raise TypeError("this schedule reads a direct affine representation")
    if representation.coefficients.scale_dtype != DType.BF16:
        raise ValueError("this schedule requires BF16 affine coefficients")
    return representation, representation.coefficients


def partitions(m, n, k):
    parts = min(max(1, 512 // (((m + 31) // 32) * ((n + 31) // 32))), k // 64)
    while k % (parts * 64):
        parts -= 1
    return parts
