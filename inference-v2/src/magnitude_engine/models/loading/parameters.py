"""Shared resident parameter construction, independent of neural family semantics."""

from collections.abc import Callable, Mapping
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.artifacts.layouts import LogicalTensor
from magnitude_engine.artifacts.materialization import (
    ModelAllocation,
    ResidentMaterializer,
    ResidentTensors,
)
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding, ResidentEmbedding
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader

from ..embeddings.contracts import EmbeddingLookup
from ..experts.computation import (
    ExpertWeights,
    GatedExpertMath,
    QuantizedProjection,
    ResidentExperts,
)
from .validation import configure_affine_modules, validate_parameters


def affine_encodings(
    tensors: Mapping[str, LogicalTensor], settings: Mapping[str, Any], *, prefix: str
) -> dict[str, AffineEncoding]:
    encodings = {}
    for name, tensor in tensors.items():
        if not name.endswith(".scales"):
            continue
        module = name.removesuffix(".scales")
        selected = settings.get(prefix + module, settings.get(module, settings))
        if selected.get("mode", "affine") != "affine":
            raise ValueError("resident affine binding cannot interpret a non-affine encoding")
        weight, bias = tensors.get(module + ".weight"), tensors.get(module + ".biases")
        if (
            weight is None
            or weight.dtype != "U32"
            or bias is None
            or tensor.dtype not in ("F16", "BF16", "F32")
            or bias.dtype != tensor.dtype
        ):
            raise ValueError(f"invalid affine tensor components for {module}")
        encodings[module + ".weight"] = AffineEncoding(selected["bits"], selected["group_size"])
    return encodings


def load_resident_parameters(
    model: nn.Module,
    tensors: dict[str, LogicalTensor],
    encodings: dict[str, AffineEncoding],
    *,
    budget: MemoryBudget,
    reader: PositionalReader,
    owner: str,
    excluded: frozenset[str] = frozenset(),
) -> ModelAllocation:
    configure_affine_modules(model, encodings)
    validate_parameters(model, {name: tensor.shape for name, tensor in tensors.items()})
    return materialize_parameters(
        model,
        tensors,
        budget=budget,
        reader=reader,
        owner=owner,
        excluded=excluded,
    )


def materialize_parameters(
    model: nn.Module,
    tensors: dict[str, LogicalTensor],
    *,
    budget: MemoryBudget,
    reader: PositionalReader,
    owner: str,
    excluded: frozenset[str] = frozenset(),
) -> ModelAllocation:
    """Install a validated parameter partition after geometry/ownership binding."""
    if not excluded <= tensors.keys():
        raise ValueError("excluded tensors are outside the validated model layout")
    weights: ResidentTensors = ResidentMaterializer(budget, reader, owner=owner).materialize(
        {name: tensor for name, tensor in tensors.items() if name not in excluded}
    )
    allocation = ModelAllocation({owner: weights})
    try:
        model.load_weights(list(weights.arrays.items()), strict=not excluded)
    except BaseException:
        allocation.close()
        raise
    return allocation


def resident_embedding(module: Any) -> tuple[EmbeddingLookup, mx.Dtype]:
    if isinstance(module, nn.QuantizedEmbedding):
        if module.mode != "affine" or module.biases is None:
            raise ValueError("resident affine embedding requires scales and biases")
        return ResidentAffineEmbedding(
            module.weight,
            module.scales,
            module.biases,
            AffineEncoding(module.bits, module.group_size),
        ), module.scales.dtype
    return ResidentEmbedding(module.weight), module.weight.dtype


def resident_experts(
    up: Any, gate: Any, down: Any, activation: Callable[[mx.array, mx.array], mx.array]
) -> ResidentExperts:
    projections = []
    for module in (up, gate, down):
        if module.mode != "affine" or module.get("bias") is not None:
            raise ValueError("unsupported expert projection encoding or additive bias")
        projections.append(
            QuantizedProjection(
                module.weight,
                module.scales,
                module.biases,
                AffineEncoding(module.bits, module.group_size),
            )
        )
    return ResidentExperts(ExpertWeights(*projections), GatedExpertMath(activation))
