"""The one decision that turns a stored weight into a resident representation.

Nothing else in the engine chooses a layout. Compact quantized weights are
relaid into canonical bytes; affine planes keep their coefficient width; dense values
are converted according to their declared transform. The result is recorded on
the resident weight and appears in run records.
"""

from __future__ import annotations

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.descriptor import (
    Stored,
    StoredAffinePlanes,
    StoredDense,
    StoredQuantized,
)
from magnitude_engine.weights.representation import (
    Affine,
    Code,
    Dense,
    DirectCoefficients,
    Representation,
)


def resident_representation(
    stored: Stored, shape: tuple[int, ...], capability: Capability
) -> Representation:
    if isinstance(stored, StoredAffinePlanes):
        if stored.bits != 4:
            raise ValueError("affine plane upload supports a 4-bit low plane")
        return Affine(
            code=Code(stored.bits),
            group=stored.group,
            coefficients=DirectCoefficients(stored.scales.dtype, stored.biases.dtype),
        )
    if isinstance(stored, StoredDense):
        # A container's floating tensor becomes the dense FP32 its consumers
        # read; the conversion itself belongs to residency, not to this choice.
        return Dense(DType.F32)
    if not isinstance(stored, StoredQuantized):
        raise TypeError("unknown stored weight layout")
    return stored.representation
