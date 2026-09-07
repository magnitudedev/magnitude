"""Parallel projections share an input and may share one packed matrix execution."""

from typing import cast

import mlx.core as mx
import mlx.nn as nn


class ParallelProjections(nn.Module):
    def __init__(self, parts: tuple[nn.Module, ...], packed: nn.Module | None = None):
        super().__init__()
        self.sizes = tuple(cast(mx.array, part.weight).shape[-2] for part in parts)
        self.operations = (packed,) if packed is not None else parts
        self.packed = packed is not None

    def __call__(self, hidden: mx.array) -> tuple[mx.array, ...]:
        if not self.packed:
            return tuple(part(hidden) for part in self.operations)
        values = self.operations[0](hidden)
        offsets, current = [], 0
        for size in self.sizes[:-1]:
            current += size
            offsets.append(current)
        return tuple(mx.split(values, offsets, axis=-1))
