"""Model-wide numerical families, independent of target kernel selection.

All families retain FP32 reductions, delta state and logits. Compact matrix/KV
storage and residual accumulation are distinct roles. Arithmetic identities keep
checkpoint and reference compatibility explicit.
"""

from enum import StrEnum

from magnitude_engine.platform.execution import DType


class NumericalFamily(StrEnum):
    REFERENCE_F32 = "reference_f32"
    NATIVE_BF16 = "native_bf16"
    MIXED_BF16 = "mixed_bf16"
    MIXED_BF16_F32_RESIDUAL = "mixed_bf16_f32_residual"

    @property
    def activation(self) -> DType:
        return DType.F32 if self is NumericalFamily.REFERENCE_F32 else DType.BF16

    @property
    def residual(self) -> DType:
        return (
            DType.BF16
            if self in (NumericalFamily.MIXED_BF16, NumericalFamily.NATIVE_BF16)
            else DType.F32
        )

    @property
    def native_rounding(self) -> bool:
        return self is NumericalFamily.NATIVE_BF16

    @property
    def recurrent_activation(self) -> DType:
        return DType.BF16 if self.native_rounding else DType.F32


def floating(dtype: DType) -> None:
    if dtype not in (DType.F32, DType.BF16):
        raise ValueError("numerical operands require a supported floating storage type")
