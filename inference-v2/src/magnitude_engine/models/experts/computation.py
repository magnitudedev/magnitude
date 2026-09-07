"""Expert math is shared by resident and streamed storage implementations."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwiGLU

from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.components import component

from ..execution import ExecutionScope
from . import metal


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


def affine_mlp(layer) -> ExpertWeights | None:
    projections = []
    for name in ("up_proj", "gate_proj", "down_proj"):
        p = getattr(layer, name, None)
        if (
            not isinstance(p, nn.QuantizedLinear)
            or p.mode != "affine"
            or "bias" in p
            or p.bits not in (2, 4, 8)
            or p.biases is None
        ):
            return None
        projections.append(
            QuantizedProjection(p.weight, p.scales, p.biases, AffineEncoding(p.bits, p.group_size))
        )
    return ExpertWeights(*projections)


@dataclass(frozen=True)
class GatedExpertMath:
    """The architecture supplies activation(up, gate) and routing; execution owns reduction."""

    activation: Callable[[mx.array, mx.array], mx.array]

    def apply(
        self,
        weights: ExpertWeights,
        hidden: mx.array,
        assignments: mx.array,
        scores: mx.array,
        *,
        expand_assignments: bool = False,
    ) -> mx.array:
        if hidden.shape[:-1] != assignments.shape[:-1] or assignments.shape[-1] < 1:
            raise ValueError("expert assignments must match hidden rows and have nonempty top-k")
        if scores.shape != assignments.shape or scores.dtype != hidden.dtype:
            raise ValueError("expert coefficients must match routes and hidden dtype")
        if isinstance(self.activation, SwiGLU) and metal.supported(weights, hidden, assignments):
            return metal.apply(weights, hidden, assignments, scores)
        shape = assignments.shape
        top_k = shape[-1]
        rows = hidden.reshape(-1, hidden.shape[-1])
        sorted_indices = assignments.size >= 64
        restore = None
        if sorted_indices or expand_assignments:
            flat = assignments.reshape(-1)
            order = mx.argsort(flat) if sorted_indices else mx.arange(flat.size)
            inputs = rows[order // top_k][:, None, :]
            indices = flat[order]
            restore = mx.argsort(order)
        else:
            inputs = rows[:, None, None, :]
            indices = assignments.reshape(-1, top_k)
        up = weights.up.apply(inputs, indices, sorted_indices=sorted_indices)
        gate = weights.gate.apply(inputs, indices, sorted_indices=sorted_indices)
        output = weights.down.apply(
            self.activation(up, gate), indices, sorted_indices=sorted_indices
        )
        output = output.squeeze(-2)
        if restore is not None:
            output = output[restore]
        output = output.reshape(*shape, hidden.shape[-1])
        return (output * scores[..., None]).sum(axis=-2)


@dataclass(frozen=True)
@component("MODEL:EXPERTS:MAG:RESIDENT_GATHERED")
class ResidentExperts:
    weights: ExpertWeights
    math: GatedExpertMath

    def __call__(self, hidden: mx.array, assignments: mx.array, scores: mx.array) -> mx.array:
        return self.math.apply(self.weights, hidden, assignments, scores)

    def compute(
        self, hidden: mx.array, assignments: mx.array, scores: mx.array, scope: ExecutionScope
    ) -> mx.array:
        # Resident weights require no scratch-retirement boundary. Consumers
        # determine liveness; rooting the expanded expert outputs here would keep
        # every layer's intermediates alive and force dead prefill computation.
        return self(hidden, assignments, scores)
