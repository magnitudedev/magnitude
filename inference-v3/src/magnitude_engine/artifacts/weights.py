"""Logical weight values, independent of their container and device representation."""

from enum import StrEnum

from pydantic import PositiveInt

from magnitude_engine.data import Record


class WeightTransform(StrEnum):
    IDENTITY = "identity"
    NEGATIVE_EXP = "negative_exp"


class WeightDescriptor(Record):
    name: str
    shape: tuple[PositiveInt, ...]
    transform: WeightTransform = WeightTransform.IDENTITY
