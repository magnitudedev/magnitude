"""Tensor encoding metadata, shared by every operation consuming those tensors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AffineEncoding:
    bits: int
    group_size: int

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8) or self.group_size <= 0:
            raise ValueError("unsupported affine encoding")
