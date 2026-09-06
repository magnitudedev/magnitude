from dataclasses import field

from magnitude_engine.composition import Blueprint, component
from magnitude_engine.resources.io.blueprint import PositionalReader
from magnitude_engine.resources.io.reader import PositionalReader as Reader

from .contracts import EmbeddingFactory


@component
class Resident(Blueprint[EmbeddingFactory]):
    @staticmethod
    def implementation() -> type[EmbeddingFactory]:
        from .binding import Resident

        return Resident


@component
class Streamed(Blueprint[EmbeddingFactory]):
    cache_bytes: int = 64 << 20
    max_pending: int = 2
    reader: Blueprint[Reader] = field(default_factory=PositionalReader)

    def __post_init__(self) -> None:
        if self.cache_bytes < 0 or not 1 <= self.max_pending <= 8:
            raise ValueError("invalid embedding cache or staging capacity")

    @staticmethod
    def implementation() -> type[EmbeddingFactory]:
        from .binding import Streamed

        return Streamed
