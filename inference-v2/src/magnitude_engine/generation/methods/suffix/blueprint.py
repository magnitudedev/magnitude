from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.generation.contracts import MethodFactory


@blueprint
class Suffix(Blueprint[MethodFactory]):
    minimum: int = 3
    maximum: int = 6

    def __post_init__(self) -> None:
        if self.minimum < 1 or self.maximum < self.minimum:
            raise ValueError("invalid suffix proposal lengths")

    @staticmethod
    def implementation() -> type[MethodFactory]:
        from .binding import Suffix

        return Suffix
