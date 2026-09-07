"""Resident dense and affine embedding row lookup."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine import components as c
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.components import component

from ..execution import ExecutionScope


@dataclass
@component(c.EMBEDDING, source=c.Source.MAG, variant="RESIDENT")
class ResidentEmbedding:
    weight: mx.array

    def __call__(self, rows: mx.array) -> mx.array:
        return self.weight[rows]

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        output = self(rows)
        scope.depend(output)
        return output


@dataclass
@component(c.EMBEDDING, source=c.Source.MAG, variant="RESIDENT")
class ResidentAffineEmbedding:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    encoding: AffineEncoding

    def __call__(self, rows: mx.array) -> mx.array:
        flat = rows.reshape(-1)
        output = mx.dequantize(
            self.weight[flat],
            self.scales[flat],
            self.biases[flat],
            group_size=self.encoding.group_size,
            bits=self.encoding.bits,
            mode="affine",
        )
        return output.reshape(*rows.shape, output.shape[-1])

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        output = self(rows)
        scope.depend(output)
        return output
