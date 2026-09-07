from magnitude_engine.composition import Blueprint, blueprint

from .contracts import DeltaRecurrence


@blueprint
class Delta(Blueprint[DeltaRecurrence]):
    specialize_prefill: bool = False

    @staticmethod
    def implementation() -> type[DeltaRecurrence]:
        from .metal import MetalDelta

        return MetalDelta


@blueprint
class Reference(Blueprint[DeltaRecurrence]):
    @staticmethod
    def implementation() -> type[DeltaRecurrence]:
        from .reference import DeltaReference

        return DeltaReference


@blueprint
class MLX(Blueprint[DeltaRecurrence]):
    @staticmethod
    def implementation() -> type[DeltaRecurrence]:
        from .mlx import MLXDelta

        return MLXDelta
