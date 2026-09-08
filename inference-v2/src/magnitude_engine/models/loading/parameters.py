"""Shared resident parameter construction, independent of neural family semantics."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
from magnitude_engine.components import component
from magnitude_engine.kernels.contractions.weights import ExpertWeights, QuantizedProjection
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding, ResidentEmbedding
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader

from ..embeddings.contracts import EmbeddingLookup
from ..experts.computation import GatedExpertMath, ResidentExperts
from .packing import ProjectionPack
from .validation import configure_affine_modules, validate_parameters


@dataclass
class BoundParameters:
    allocation: ModelAllocation
    projections: dict[tuple[str, ...], nn.Module]

    def close(self) -> None:
        self.projections.clear()
        self.allocation.close()


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


@component("MODEL:LOADING:MAG:RESIDENT")
def load_resident_parameters(
    model: nn.Module,
    tensors: dict[str, LogicalTensor],
    encodings: dict[str, AffineEncoding],
    *,
    budget: MemoryBudget,
    reader: PositionalReader,
    owner: str,
    excluded: frozenset[str] = frozenset(),
) -> BoundParameters:
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
    packs: tuple[ProjectionPack, ...] = (),
    encodings: dict[str, AffineEncoding] | None = None,
) -> BoundParameters:
    """Install a validated parameter partition after geometry/ownership binding."""
    if not excluded <= tensors.keys():
        raise ValueError("excluded tensors are outside the validated model layout")
    selected = {name: tensor for name, tensor in tensors.items() if name not in excluded}
    accepted = []
    for pack in packs:
        plan = pack.plan(selected, encodings or {})
        if plan is None:
            continue
        for name in pack.names:
            for suffix in ("weight", "scales", "biases", "bias"):
                selected.pop(name + "." + suffix, None)
        selected.update(plan)
        accepted.append((pack, plan))
    weights: ResidentTensors = ResidentMaterializer(budget, reader, owner=owner).materialize(
        selected
    )
    allocation = ModelAllocation({owner: weights})
    try:
        arrays = dict(weights.arrays)
        projections = {}
        for pack, plan in accepted:
            projections[pack.names] = pack.module(weights.arrays, encodings or {})
            arrays.update(pack.views(weights.arrays, tensors))
            for name in plan:
                del arrays[name]
        model.load_weights(list(arrays.items()), strict=not excluded)
    except BaseException:
        allocation.close()
        raise
    return BoundParameters(allocation, projections)


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
