"""Resident dense and affine embedding row lookup."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.artifacts.quantization import AffineEncoding

from ..execution import ExecutionScope


@dataclass
class ResidentEmbedding:
    weight: mx.array

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        output = self.weight[rows]
        scope.depend(output)
        return output


@dataclass
class ResidentAffineEmbedding:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    encoding: AffineEncoding

    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array:
        flat = rows.reshape(-1)
        output = mx.dequantize(
            self.weight[flat],
            self.scales[flat],
            self.biases[flat],
            group_size=self.encoding.group_size,
            bits=self.encoding.bits,
            mode="affine",
        )
        output = output.reshape(*rows.shape, output.shape[-1])
        scope.depend(output)
        return output
