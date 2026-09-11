"""The precision axis: storage dtype per role, and one rounding mode.

Reductions, delta state, decay, logits and attention statistics are always FP32
and are therefore not fields. A kernel that rounds differently between modes
branches on ``precision.rounding`` while building its TIR; a rounding mode never
selects a different kernel factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from magnitude_engine.platform.execution import DType


def floating(dtype: DType) -> None:
    if dtype not in (DType.F32, DType.BF16):
        raise ValueError("numerical operands require a supported floating storage type")


class Rounding(StrEnum):
    FP32_INTERNAL = "fp32_internal"
    NATIVE_BF16 = "native_bf16"


@dataclass(frozen=True)
class Precision:
    activation: DType
    """Matrix inputs, elementwise results, attention queries."""

    residual: DType
    """Residual stream and readout input."""

    recurrent: DType
    """Recurrent q/k/v/beta and the mixed recurrent output."""

    kv: DType
    """KV store entries."""

    rounding: Rounding

    def __post_init__(self):
        for role in (self.activation, self.residual, self.recurrent, self.kv):
            floating(role)
        if not isinstance(self.rounding, Rounding):
            raise TypeError("precision requires an explicit rounding mode")


REFERENCE_F32 = Precision(
    activation=DType.F32,
    residual=DType.F32,
    recurrent=DType.F32,
    kv=DType.F32,
    rounding=Rounding.FP32_INTERNAL,
)
MIXED_BF16 = Precision(
    activation=DType.BF16,
    residual=DType.BF16,
    recurrent=DType.F32,
    kv=DType.BF16,
    rounding=Rounding.FP32_INTERNAL,
)
MIXED_BF16_F32_RESIDUAL = Precision(
    activation=DType.BF16,
    residual=DType.F32,
    recurrent=DType.F32,
    kv=DType.BF16,
    rounding=Rounding.FP32_INTERNAL,
)
NATIVE_BF16 = Precision(
    activation=DType.BF16,
    residual=DType.BF16,
    recurrent=DType.BF16,
    kv=DType.BF16,
    rounding=Rounding.NATIVE_BF16,
)

PRESETS: dict[str, Precision] = {
    "reference_f32": REFERENCE_F32,
    "mixed_bf16": MIXED_BF16,
    "mixed_bf16_f32_residual": MIXED_BF16_F32_RESIDUAL,
    "native_bf16": NATIVE_BF16,
}


def preset(name: str) -> Precision:
    if name not in PRESETS:
        raise ValueError(f"unknown precision preset {name!r}")
    return PRESETS[name]


def __getattr__(name: str):
    """Reach the TIR definition of NATIVE_BF16 rounding without importing it.

    ``native_sigmoid`` and ``native_decay`` belong beside the mode they define,
    but they are TIR macros. Blueprint inspection reads ``Precision`` as data
    and must not load a compiler, so the macros are resolved on first use.
    """
    if name in ("native_sigmoid", "native_decay"):
        from magnitude_engine.kernels import rounding

        return getattr(rounding, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
