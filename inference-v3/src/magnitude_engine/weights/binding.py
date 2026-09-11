"""The one decision that turns a stored weight into a resident representation.

Nothing else in the engine chooses a layout. The inputs are the stored layout,
the logical shape, and the endpoint capability; the result is recorded on the
resident weight and appears in run records.
"""

from __future__ import annotations

import math

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.descriptor import (
    Stored,
    StoredAffinePlanes,
    StoredBlocks,
    StoredDense,
)
from magnitude_engine.weights.representation import (
    Blocked,
    Dense,
    Encoding,
    PlanarAffine,
    Representation,
)

# Repacking a K-quant matrix into planes pays for itself only where a subgroup
# can consume a whole 512-coordinate fold; below that the container layout is
# read in place.
_REPACKED = {
    Encoding.Q4_K: PlanarAffine(
        bits=4, high_bits=0, group=32, coefficient_dtype=DType.F32, signed=False, has_bias=True
    ),
    Encoding.Q5_K: PlanarAffine(
        bits=4, high_bits=1, group=32, coefficient_dtype=DType.F32, signed=False, has_bias=True
    ),
    Encoding.Q6_K: PlanarAffine(
        bits=4, high_bits=2, group=16, coefficient_dtype=DType.F32, signed=True, has_bias=False
    ),
}


def resident_representation(
    stored: Stored, shape: tuple[int, ...], capability: Capability
) -> Representation:
    if isinstance(stored, StoredAffinePlanes):
        if stored.bits != 4:
            raise ValueError("affine plane upload supports a 4-bit low plane")
        return PlanarAffine(
            bits=stored.bits,
            high_bits=0,
            group=stored.group,
            coefficient_dtype=stored.scales.dtype,
            signed=False,
            has_bias=True,
        )
    if isinstance(stored, StoredDense):
        # A container's floating tensor becomes the dense FP32 its consumers
        # read; the conversion itself belongs to residency, not to this choice.
        return Dense(DType.F32)
    if not isinstance(stored, StoredBlocks):
        raise TypeError("unknown stored weight layout")
    if stored.encoding == Encoding.F32:
        return Dense(DType.F32)
    repacked = _REPACKED.get(stored.encoding)
    if (
        repacked is not None
        and capability.subgroup_width == 32
        and len(shape) == 2
        and shape[1] % 512 == 0
        and math.prod(shape) % repacked.group == 0
    ):
        return repacked
    return Blocked(stored.encoding)
