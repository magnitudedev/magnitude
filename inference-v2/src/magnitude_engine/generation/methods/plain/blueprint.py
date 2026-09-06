from magnitude_engine.composition import Blueprint, component
from magnitude_engine.generation.contracts import MethodFactory


@component
class Plain(Blueprint[MethodFactory]):
    @staticmethod
    def implementation() -> type[MethodFactory]:
        from .binding import Plain

        return Plain
