"""Continuous decode rounds and FIFO prefill, sharing completed execution time."""

from dataclasses import dataclass, field
from math import isfinite

from magnitude_engine.components import component

from .contracts import CompletedService, Runnable, Schedule, Scheduler, Service


@dataclass
@component("SCHEDULING:SERVICE:MAG:TIME_SHARING")
class TimeShared(Scheduler):
    max_active: int = 8
    max_queued: int = 128
    prefill_tokens: int = 512
    decode_tokens: int = 4
    prefill_stall_seconds: float | None = None
    decode_share: float = 0.5
    _decode_debt_ns: float = field(default=0.0, init=False)
    _contended: bool = field(default=False, init=False)
    _prefill_run_ns: int = field(default=0, init=False)
    _prefill_rate: float | None = field(default=None, init=False)
    _pending: Schedule | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if (
            any(
                type(n) is not int or n < 1
                for n in (self.max_active, self.max_queued, self.prefill_tokens, self.decode_tokens)
            )
            or self.max_active > 64
            or (
                self.prefill_stall_seconds is not None
                and (not isfinite(self.prefill_stall_seconds) or self.prefill_stall_seconds <= 0)
            )
            or not isfinite(self.decode_share)
            or not 0 < self.decode_share < 1
        ):
            raise ValueError("invalid scheduler capacity or time-sharing policy")

    def select(self, rows: tuple[Runnable, ...]) -> Schedule | None:
        if self._pending is not None:
            raise RuntimeError("complete scheduled service before selecting more work")
        decoding = tuple(row for row in rows if not row.prefill_remaining and row.output_credit)
        waiting = tuple(row for row in rows if row.prefill_remaining and row.output_credit)
        contended = bool(decoding) and bool(waiting)
        first_round = contended and not self._contended
        if not contended or first_round:
            self._decode_debt_ns = 0.0
            self._prefill_run_ns = 0
        self._contended = contended

        interruption_ns = (
            None
            if self.prefill_stall_seconds is None
            else max(1, int(self.prefill_stall_seconds * 1e9))
        )
        if decoding and (
            not waiting
            or first_round
            or self._decode_debt_ns > 0
            or (interruption_ns is not None and self._prefill_run_ns >= interruption_ns)
        ):
            plan = Schedule(
                "decode",
                tuple(
                    Service(row.identity, min(row.output_credit, self.decode_tokens))
                    for row in decoding
                ),
                interruption_ns if contended else None,
            )
        elif waiting:
            oldest = waiting[0]
            waiting = tuple(
                row
                for row in waiting
                if row is oldest
                or (oldest.prefill_group is not None and row.prefill_group == oldest.prefill_group)
            )
            count = self.prefill_tokens
            if decoding and interruption_ns is not None and self._prefill_rate is not None:
                remaining_seconds = (interruption_ns - self._prefill_run_ns) / 1e9
                count = min(count, max(1, int(self._prefill_rate * remaining_seconds)))
            # Share one aggregate prompt-token budget among admitted FIFO rows.
            # Equal chunk widths expose batching without padding or fabricated KV.
            selected = waiting[:count]
            width = max(1, count // len(selected))
            plan = Schedule(
                "prefill",
                tuple(Service(row.identity, min(row.prefill_remaining, width)) for row in selected),
            )
        else:
            return None
        self._pending = plan
        return plan

    def observe(self, service: CompletedService) -> None:
        plan = self._pending
        if plan is None or service.phase != plan.phase:
            raise ValueError("service feedback must match the selected phase")
        if (
            type(service.elapsed_ns) is not int
            or service.elapsed_ns < 0
            or type(service.input_tokens) is not int
            or service.input_tokens < 0
        ):
            raise ValueError("service duration and input count must be nonnegative integers")
        self._pending = None
        if self._contended:
            if service.phase == "prefill":
                self._prefill_run_ns += service.elapsed_ns
                self._decode_debt_ns += (
                    service.elapsed_ns * self.decode_share / (1 - self.decode_share)
                )
            else:
                self._prefill_run_ns = 0
                # A mandatory decode interruption may overserve the desired share.
                # Keep at most this round's credit rather than banking unbounded
                # service against future, potentially much cheaper, decode rounds.
                self._decode_debt_ns = max(
                    -service.elapsed_ns, self._decode_debt_ns - service.elapsed_ns
                )
        if service.phase == "prefill" and service.input_tokens and service.elapsed_ns:
            rate = service.input_tokens * 1e9 / service.elapsed_ns
            self._prefill_rate = (
                rate if self._prefill_rate is None else 0.7 * self._prefill_rate + 0.3 * rate
            )

    def reset(self) -> None:
        if self._pending is not None:
            raise RuntimeError("cannot reset a scheduler with outstanding service")
        self._decode_debt_ns = 0.0
        self._contended = False
        self._prefill_run_ns = 0
        self._prefill_rate = None
