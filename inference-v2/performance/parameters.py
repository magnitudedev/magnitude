"""Tensor facts from explicitly bound operands and supported upstream containers."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.components import (
    MatrixFacts,
    NeuralParameters,
    TensorFacts,
)


def tensors(value: object) -> dict[str, mx.array]:
    """Supported parameter containers, never a walk through arbitrary object fields."""
    if value is None:
        return {}
    if isinstance(value, mx.array):
        return {"value": value}
    if inspect.ismethod(value) and isinstance(value.__self__, nn.Module):
        return tensors(value.__self__)
    if isinstance(value, nn.Module):
        return tensors(value.parameters())
    if isinstance(value, dict):
        return {
            f"{k}.{name}": a for k, child in value.items() for name, a in tensors(child).items()
        }
    if isinstance(value, (tuple, list)):
        return {
            f"{i}.{name}": a for i, child in enumerate(value) for name, a in tensors(child).items()
        }
    from magnitude_engine.models.architectures.qwen35.mtp.loading import AttentionStep
    from magnitude_engine.models.experts.computation import QuantizedProjection

    if isinstance(value, QuantizedProjection):
        return {"weight": value.weight, "scales": value.scales, "biases": value.biases}
    if isinstance(value, AttentionStep):
        return tensors(value.layer)
    raise TypeError(f"no declared parameter adapter for {type(value).__qualname__}")


def projection_shape(value: object) -> tuple[int, int]:
    if inspect.ismethod(value) and isinstance(value.__self__, nn.Module):
        value = value.__self__
    if isinstance(value, (nn.Linear, nn.Embedding)):
        return value.weight.shape[-1], value.weight.dtype.size
    if isinstance(value, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        return value.weight.shape[-1] * 32 // value.bits, value.scales.dtype.size
    from magnitude_engine.models.experts.computation import QuantizedProjection

    if isinstance(value, QuantizedProjection):
        return value.weight.shape[-1] * 32 // value.encoding.bits, value.scales.dtype.size
    raise TypeError(f"no projection geometry adapter for {type(value).__qualname__}")


def materialize(parameters: NeuralParameters, operands: Mapping[str, object]):
    from magnitude_engine.models.experts.computation import QuantizedProjection

    arrays, matrices, bound_tensors = {}, [], {}
    configuration = dict(parameters.settings)
    for role, operand in operands.items():
        for name, value in tensors(operand).items():
            key = role + "." + name
            arrays[key] = TensorFacts(
                identity=key, shape=value.shape, bytes=value.nbytes, dtype=str(value.dtype)
            )
            bound_tensors[key] = value
        matrix = operand.__self__ if inspect.ismethod(operand) else operand
        if isinstance(matrix, (nn.RMSNorm, nn.LayerNorm)):
            configuration[role + ".eps"] = matrix.eps
        if isinstance(matrix, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            configuration[role + ".bits"] = matrix.bits
            configuration[role + ".group_size"] = matrix.group_size
            configuration[role + ".mode"] = matrix.mode
        if isinstance(matrix, (nn.Linear, nn.QuantizedLinear, QuantizedProjection)):
            width, _ = projection_shape(matrix)
            matrices.append(
                MatrixFacts(
                    identity=role,
                    input_width=width,
                    output_width=matrix.weight.shape[-2],
                    experts=matrix.weight.shape[0] if matrix.weight.ndim == 3 else None,
                )
            )
    return parameters.model_copy(
        update={
            "arrays": {**parameters.arrays, **arrays},
            "matrices": tuple(matrices),
            "settings": configuration,
        }
    ), bound_tensors


def projection_output_width(value: object) -> int:
    if inspect.ismethod(value):
        value = value.__self__
    if isinstance(value, (nn.Linear, nn.QuantizedLinear, nn.Embedding, nn.QuantizedEmbedding)):
        return value.weight.shape[-2]
    raise TypeError(f"no projection output adapter for {type(value).__qualname__}")
