from magnitude_engine.composition import Blueprint, component

from .contracts import DeltaRecurrence


@component
class Delta(Blueprint[DeltaRecurrence]):
    specialize_prefill: bool = False

    @staticmethod
    def implementation() -> type[DeltaRecurrence]:
        from .metal import MetalDelta

        return MetalDelta


@component
class Reference(Blueprint[DeltaRecurrence]):
    @staticmethod
    def implementation() -> type[DeltaRecurrence]:
        from .reference import DeltaReference

        return DeltaReference


@component
class MLX(Blueprint[DeltaRecurrence]):
    @staticmethod
    def implementation() -> type[DeltaRecurrence]:
        from .mlx import MLXDelta

        return MLXDelta
