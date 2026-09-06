from magnitude_engine.composition import Blueprint, component
from magnitude_engine.generation.contracts import MethodFactory
from magnitude_engine.models.executor.contracts import ExecutorFactory


@component
class MTP(Blueprint[MethodFactory]):
    drafter: Blueprint[ExecutorFactory]
    max_draft_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_draft_tokens is not None and self.max_draft_tokens < 1:
            raise ValueError("draft allowance must be positive")

    @staticmethod
    def implementation() -> type[MethodFactory]:
        from .binding import MTP

        return MTP
