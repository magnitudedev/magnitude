"""Encoded operands shared by contraction plans and model bindings."""

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.artifacts.quantization import AffineEncoding


@dataclass(frozen=True)
class QuantizedProjection:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    encoding: AffineEncoding

    def apply(self, inputs: mx.array, assignments: mx.array, *, sorted_indices: bool) -> mx.array:
        return mx.gather_qmm(
            inputs,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=assignments,
            transpose=True,
            group_size=self.encoding.group_size,
            bits=self.encoding.bits,
            mode="affine",
            sorted_indices=sorted_indices,
        )


@dataclass(frozen=True)
class ExpertWeights:
    up: QuantizedProjection
    gate: QuantizedProjection
    down: QuantizedProjection
