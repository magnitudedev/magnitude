"""Handwritten Metal declarations and compatible MLX compilation."""

from .core import metal
from .core.compiler import artifact, compile, explain
from .core.declaration import kernel

__all__ = ["kernel", "compile", "explain", "artifact", "metal"]
