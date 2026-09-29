"""Borrowed stateless transforms used to assemble neural programs."""

from collections.abc import Callable
from typing import Protocol

import mlx.core as mx

Transform = Callable[[mx.array], mx.array]


class PositionTransform(Protocol):
    def __call__(self, x: mx.array, *, offset: int | mx.array) -> mx.array: ...
