from dataclasses import field

from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.resources.io.blueprint import PositionalReader
from magnitude_engine.resources.io.reader import PositionalReader as Reader

from .contracts import ExpertFactory


@blueprint
class Resident(Blueprint[ExpertFactory]):
    @staticmethod
    def implementation() -> type[ExpertFactory]:
        from .binding import Resident

        return Resident


@blueprint
class Streamed(Blueprint[ExpertFactory]):
    slots: int = 8
    reader: Blueprint[Reader] = field(default_factory=PositionalReader)

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError("expert bank capacity must be positive")

    @staticmethod
    def implementation() -> type[ExpertFactory]:
        from .binding import Streamed

        return Streamed
