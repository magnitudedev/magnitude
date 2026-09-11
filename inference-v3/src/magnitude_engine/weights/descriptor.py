"""Logical weight values, and what a container yields for one of them.

A descriptor names a role the model asked for. A ``Stored`` value is what a
format found on disk for that role: a byte source, an offset, and the layout the
bytes are in. Neither says anything about how a kernel will read the weight;
that is a ``Representation``, decided at residency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import PositiveInt

from magnitude_engine.data import Record
from magnitude_engine.platform.execution import DType
from magnitude_engine.platform.storage import ByteSource
from magnitude_engine.weights.representation import EncodedBlocks, HierarchicalAffine


class WeightTransform(StrEnum):
    IDENTITY = "identity"
    NEGATIVE_EXP = "negative_exp"


class WeightDescriptor(Record):
    name: str
    shape: tuple[PositiveInt, ...]
    transform: WeightTransform = WeightTransform.IDENTITY


@dataclass(frozen=True)
class StoredBlocks:
    """A neutral block layout, exactly as the container holds it."""

    layout: EncodedBlocks | HierarchicalAffine
    source: ByteSource
    offset: int


@dataclass(frozen=True)
class StoredDense:
    dtype: DType
    source: ByteSource
    offset: int
    nbytes: int


@dataclass(frozen=True)
class StoredAffinePlanes:
    """Affine codes and per-group coefficients held as three separate tensors."""

    bits: int
    group: int
    codes: StoredDense
    scales: StoredDense
    biases: StoredDense


type Stored = StoredBlocks | StoredDense | StoredAffinePlanes
