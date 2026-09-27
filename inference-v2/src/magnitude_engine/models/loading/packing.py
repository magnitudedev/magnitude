"""Read compatible projection groups directly into their final execution layout."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.artifacts.layouts import LogicalTensor
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.tensors import FileSlice


def concatenate(name: str, tensors: tuple[LogicalTensor, ...], axis: int) -> LogicalTensor:
    """Concatenate logical storage extents, including interleaved expert rows."""
    if not tensors:
        raise ValueError("packing requires nonempty tensor inputs")
    first = tensors[0]
    if not 0 <= axis < len(first.shape) or any(
        t.dtype != first.dtype
        or len(t.shape) != len(first.shape)
        or t.shape[:axis] != first.shape[:axis]
        or t.shape[axis + 1 :] != first.shape[axis + 1 :]
        for t in tensors
    ):
        raise ValueError("packed tensor geometry or dtype differs")
    outer = prod(first.shape[:axis])
    pieces: list[FileSlice] = []
    for row in range(outer):
        for tensor in tensors:
            width = tensor.nbytes // outer
            start, stop = row * width, (row + 1) * width
            cursor = 0
            for piece in tensor.pieces:
                lo, hi = max(start, cursor), min(stop, cursor + piece.size)
                if lo < hi:
                    pieces.append(piece.slice(lo - cursor, hi - lo))
                cursor += piece.size
                if cursor >= stop:
                    break
    shape = (*first.shape[:axis], sum(t.shape[axis] for t in tensors), *first.shape[axis + 1 :])
    return LogicalTensor(name, shape, first.dtype, tuple(pieces))


@dataclass(frozen=True)
class ProjectionPack:
    """Artifact projection names sharing inputs, encoding and output-axis packing."""

    names: tuple[str, ...]

    def plan(self, tensors: dict[str, LogicalTensor], encodings: dict[str, AffineEncoding]):
        if len(self.names) < 2 or len(set(self.names)) != len(self.names):
            raise ValueError("a projection pack needs distinct projection names")
        selected = [encodings.get(name + ".weight") for name in self.names]
        if any(encoding != selected[0] for encoding in selected):
            return None
        encoding = selected[0]
        components = ("weight", "scales", "biases") if encoding is not None else ("weight",)
        # Additive biases must be uniformly present. Quantization offsets are separate.
        biases = [name + ".bias" in tensors for name in self.names]
        if any(biases) != all(biases):
            return None
        if all(biases):
            components += ("bias",)
        weights = [tensors[name + ".weight"] for name in self.names]
        if any(len(w.shape) != 2 for w in weights):
            raise ValueError("ordinary projection packing requires matrix weights")
        packed = {}
        for suffix in components:
            key = self.names[0] + ".packed." + suffix
            try:
                packed[key] = concatenate(
                    key, tuple(tensors[name + "." + suffix] for name in self.names), 0
                )
            except ValueError:
                return None
        return packed

    def views(self, arrays: dict[str, mx.array], tensors: dict[str, LogicalTensor]):
        """Restore named parameter views for construction, without another allocation."""
        result = {}
        components = [
            key.removeprefix(self.names[0] + ".packed.")
            for key in arrays
            if key.startswith(self.names[0] + ".packed.")
        ]
        for suffix in components:
            source = arrays[self.names[0] + ".packed." + suffix]
            cursor = 0
            for name in self.names:
                size = tensors[name + "." + suffix].shape[0]
                result[name + "." + suffix] = source[cursor : cursor + size]
                cursor += size
        return result

    def module(
        self, arrays: dict[str, mx.array], encodings: dict[str, AffineEncoding]
    ) -> nn.Module:
        encoding = encodings.get(self.names[0] + ".weight")
        values = {
            key.removeprefix(self.names[0] + ".packed."): value
            for key, value in arrays.items()
            if key.startswith(self.names[0] + ".packed.")
        }
        if encoding is None:
            module = nn.Linear.__new__(nn.Linear)
        else:
            module = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
        nn.Module.__init__(module)
        if encoding is not None:
            module.group_size = encoding.group_size
            module.bits = encoding.bits
            module.mode = "affine"
        for name, value in values.items():
            module[name] = value
        module.eval()
        return module
