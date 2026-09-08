"""Composable MLX computations backed by typed handwritten Metal."""

from .core.computation import computation
from .core.operation import operation

__all__ = ["computation", "operation"]
