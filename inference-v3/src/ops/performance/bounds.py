"""Mathematical quantity intervals, not implementation latency predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Bounds:
    lower: float
    upper: float

    def __post_init__(self):
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("invalid mathematical interval")

    @classmethod
    def exact(cls, value):
        return cls(value, value)

    @classmethod
    def unknown(cls):
        return cls(0, math.inf)

    @property
    def fixed(self):
        return self.lower == self.upper

    def __add__(self, other):
        if not isinstance(other, Bounds):
            other = Bounds.exact(other)
        return Bounds(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Bounds):
            other = Bounds.exact(other)
        endpoints = [0 if left == 0 or right == 0 else left * right
                     for left in (self.lower, self.upper) for right in (other.lower, other.upper)]
        return Bounds(min(endpoints), max(endpoints))

    __rmul__ = __mul__


ZERO = Bounds.exact(0)
