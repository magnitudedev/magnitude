from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.generation.contracts import MethodFactory


@blueprint
class Plain(Blueprint[MethodFactory]):
    @staticmethod
    def implementation() -> type[MethodFactory]:
        from .binding import Plain

        return Plain
