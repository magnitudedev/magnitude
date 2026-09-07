from magnitude_engine.composition import Blueprint, blueprint

from .contracts import Scheduler


@blueprint
class TimeShared(Blueprint[Scheduler]):
    max_active: int = 8
    max_queued: int = 128
    prefill_tokens: int = 512
    decode_tokens: int = 4
    prefill_stall_seconds: float | None = None
    decode_share: float = 0.5

    def __post_init__(self) -> None:
        if (
            not 1 <= self.max_active <= 64
            or min(self.max_queued, self.prefill_tokens, self.decode_tokens) < 1
            or (
                self.prefill_stall_seconds is not None
                and not 0 < self.prefill_stall_seconds < float("inf")
            )
            or not 0 < self.decode_share < 1
        ):
            raise ValueError("invalid scheduler capacity")

    @staticmethod
    def implementation() -> type[Scheduler]:
        from .time_shared import TimeShared

        return TimeShared
