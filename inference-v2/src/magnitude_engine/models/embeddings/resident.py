"""Resident dense and affine embedding row lookup."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.components import component

from ..execution import ExecutionScope
from .metal import lookup


@dataclass
@component("MODEL:EMBEDDING:MAG:RESIDENT")
class ResidentEmbedding:
    weight: mx.array

    def __call__(self, rows: mx.array) -> mx.array:
        return self.weight[rows]

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        output = self(rows)
        scope.depend(output)
        return output


@dataclass
@component(ResidentEmbedding)
class ResidentAffineEmbedding:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    encoding: AffineEncoding

    def __call__(self, rows: mx.array) -> mx.array:
        return lookup(
            rows,
            self.weight,
            self.scales,
            self.biases,
            bits=self.encoding.bits,
            group_size=self.encoding.group_size,
        )

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        output = self(rows)
        scope.depend(output)
        return output
