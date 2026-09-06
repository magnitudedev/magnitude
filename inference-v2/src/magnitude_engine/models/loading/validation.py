"""Shared construction checks for pinned library parameter containers."""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from magnitude_engine.artifacts.layouts import LogicalTensor
from magnitude_engine.artifacts.quantization import AffineEncoding


def canonical_names(tensors: dict[str, LogicalTensor], prefix: str) -> dict[str, LogicalTensor]:
    result = {}
    for name, tensor in tensors.items():
        name = name.removeprefix(prefix)
        if name in result:
            raise ValueError(f"duplicate canonical tensor name: {name}")
        result[name] = tensor
    return result


def configure_affine_modules(model: nn.Module, encodings: dict[str, AffineEncoding]) -> None:
    def policy(path, module):
        encoding = encodings.get(path + ".weight")
        if encoding is None:
            return False
        if not hasattr(module, "to_quantized"):
            raise ValueError(f"encoding plan names a non-quantizable module: {path}")
        return {"bits": encoding.bits, "group_size": encoding.group_size, "mode": "affine"}

    nn.quantize(model, class_predicate=policy)


def validate_parameters(model: nn.Module, shapes: dict[str, tuple[int, ...]]) -> None:
    expected = {
        name: cast(mx.array, value).shape for name, value in tree_flatten(model.parameters())
    }
    missing, extra = expected.keys() - shapes.keys(), shapes.keys() - expected.keys()
    mismatch = [name for name in expected.keys() & shapes.keys() if expected[name] != shapes[name]]
    if missing or extra or mismatch:
        raise ValueError(
            f"artifact parameter layout differs: missing={sorted(missing)}, "
            f"extra={sorted(extra)}, shapes={sorted(mismatch)}"
        )


def encoded_shapes(
    tensors: dict[str, LogicalTensor], encodings: dict[str, AffineEncoding]
) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for name, tensor in tensors.items():
        encoding = encodings.get(name)
        if encoding is None:
            shapes[name] = tensor.shape
        else:
            shapes[name] = (*tensor.shape[:-1], tensor.shape[-1] * encoding.bits // 32)
            for suffix in (".scales", ".biases"):
                shapes[name[:-7] + suffix] = (
                    *tensor.shape[:-1],
                    tensor.shape[-1] // encoding.group_size,
                )
    return shapes
