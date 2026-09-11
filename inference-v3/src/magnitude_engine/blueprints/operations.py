"""Logical operations over one residency. Kernel selection is not exposed here.

Two runs with the same composition digest on different hardware may realize
different candidates; run records name which, so the difference is visible.
"""

from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.kernels.semantics import HeadMapping
from magnitude_engine.models.qwen35.arena import Arena as ScratchArena
from magnitude_engine.operations.attention import CausalAttention as Attention
from magnitude_engine.operations.binding import Operations as Binding
from magnitude_engine.operations.linear import ResidentLinear
from magnitude_engine.operations.recurrent import DeltaRecurrence as Recurrence
from magnitude_engine.operations.sampling import SampleSelector as Selector
from magnitude_engine.platform.execution import DeviceContext
from magnitude_engine.weights.residency import ResidentWeight as Weight
from magnitude_engine.weights.residency import Weights

__all__ = [
    "Arena",
    "CausalAttention",
    "DeltaRecurrence",
    "Linear",
    "Operations",
    "ResidentWeight",
    "SampleSelector",
]


@blueprint
class Arena(Blueprint[ScratchArena]):
    context: Blueprint[DeviceContext]
    precision: str = "native_bf16"

    @staticmethod
    def implementation():
        def build(context: DeviceContext, precision: str) -> ScratchArena:
            from magnitude_engine.kernels.precision import preset

            return ScratchArena(context, preset(precision))

        return build


@blueprint
class Operations(Blueprint[Binding]):
    weights: Blueprint[Weights]
    arena: Blueprint[ScratchArena]
    precision: str = "native_bf16"

    @staticmethod
    def implementation():
        def build(weights: Weights, arena: ScratchArena, precision: str) -> Binding:
            from magnitude_engine.kernels.precision import preset

            return Binding(weights, preset(precision), arena)

        return build


@blueprint
class CausalAttention(Blueprint[Attention]):
    context: Blueprint[DeviceContext]
    heads: int
    kv_heads: int
    width: int
    arena: Blueprint[ScratchArena]
    precision: str = "native_bf16"

    @staticmethod
    def implementation():
        def build(
            context: DeviceContext,
            heads: int,
            kv_heads: int,
            width: int,
            arena: ScratchArena,
            precision: str,
        ) -> Attention:
            from magnitude_engine.kernels.precision import preset

            return Attention(context, heads, kv_heads, width, preset(precision), arena)

        return build


@blueprint
class DeltaRecurrence(Blueprint[Recurrence]):
    context: Blueprint[DeviceContext]
    key_heads: int
    value_heads: int
    width: int
    mapping: HeadMapping
    precision: str = "native_bf16"

    @staticmethod
    def implementation():
        def build(
            context: DeviceContext,
            key_heads: int,
            value_heads: int,
            width: int,
            mapping: HeadMapping,
            precision: str,
        ) -> Recurrence:
            from magnitude_engine.kernels.precision import preset

            return Recurrence(context, key_heads, value_heads, width, mapping, preset(precision))

        return build


@blueprint
class SampleSelector(Blueprint[Selector]):
    context: Blueprint[DeviceContext]
    tile: int = 1024

    @staticmethod
    def implementation():
        return Selector


@blueprint
class ResidentWeight(Blueprint[Weight]):
    """One weight made resident on its own, for measuring a single operation."""

    weights: Blueprint[Weights]
    tensor_name: str
    shape: tuple[int, ...]

    @staticmethod
    def implementation():
        def build(weights: Weights, tensor_name: str, shape: tuple[int, ...]) -> Weight:
            from magnitude_engine.weights.descriptor import WeightDescriptor

            return weights.resident(WeightDescriptor(name=tensor_name, shape=shape))

        return build


@blueprint
class Linear(Blueprint[ResidentLinear]):
    weight: Blueprint[Weight]
    arena: Blueprint[ScratchArena]
    precision: str = "native_bf16"

    @staticmethod
    def implementation():
        def build(weight: Weight, arena: ScratchArena, precision: str) -> ResidentLinear:
            from magnitude_engine.kernels.precision import preset
            from magnitude_engine.operations.projections import ResidentProjections

            return ResidentLinear(ResidentProjections(weight, preset(precision), arena))

        return build
