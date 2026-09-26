"""Parallel projections share an input and may share one packed matrix execution."""

from typing import cast

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.kernels.contractions import linear


class QuantizedLinear(nn.QuantizedLinear):
    """Borrowed encoded weights with independent request rows during decode."""

    def __init__(self, source: nn.QuantizedLinear | nn.QuantizedEmbedding):
        nn.Module.__init__(self)
        self.weight, self.scales, self.biases = source.weight, source.scales, source.biases
        self.group_size, self.bits, self.mode = source.group_size, source.bits, source.mode
        if "bias" in source:
            self.bias = source["bias"]
        self.freeze()

    def __call__(self, inputs: mx.array) -> mx.array:
        if inputs.ndim != 3 or inputs.shape[1] > 8:
            return super().__call__(inputs)
        if inputs.size == 0:
            return mx.zeros((*inputs.shape[:-1], self.weight.shape[-2]), inputs.dtype)
        output = None
        if self.mode == "affine" and self.biases is not None:
            output = linear.apply(
                inputs,
                self.weight,
                self.scales,
                self.biases,
                bits=self.bits,
                group_size=self.group_size,
            )
        if output is None:
            # Unsupported encodings/geometries retain the same one-row operation.
            # The common short-query family shares weights without a fake bank axis.
            rows = inputs.reshape(-1, inputs.shape[-1])
            return mx.concatenate(
                [super(QuantizedLinear, self).__call__(row[None]) for row in rows]
            ).reshape(*inputs.shape[:-1], self.weight.shape[-2])
        return output + self.bias if "bias" in self else output


def bind_linear(module: nn.Module) -> nn.Module:
    return QuantizedLinear(module) if isinstance(module, nn.QuantizedLinear) else module


def bind_readout(module: nn.Module):
    if isinstance(module, nn.QuantizedEmbedding):
        return QuantizedLinear(module)
    return module.as_linear if isinstance(module, nn.Embedding) else bind_linear(module)


class ParallelProjections(nn.Module):
    def __init__(self, parts: tuple[nn.Module, ...], packed: nn.Module | None = None):
        super().__init__()
        self.sizes = tuple(cast(mx.array, part.weight).shape[-2] for part in parts)
        self.operations = (
            (bind_linear(packed),) if packed is not None else tuple(map(bind_linear, parts))
        )
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
