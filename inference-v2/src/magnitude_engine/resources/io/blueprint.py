from magnitude_engine.composition import Blueprint, blueprint

from .reader import PositionalReader as Reader


@blueprint
class PositionalReader(Blueprint[Reader]):
    workers: int = 4
    max_pending: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.workers <= 64 or not 1 <= self.max_pending <= 64:
            raise ValueError("invalid reader concurrency")

    @staticmethod
    def implementation() -> type[Reader]:
        from .reader import PositionalReader

        return PositionalReader
