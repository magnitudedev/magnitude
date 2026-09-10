"""One prepare/invoke/complete/validate/assess mechanism for every component."""

from typing import Protocol

from pydantic import Field

from magnitude_engine.platform.execution import Ticket
from magnitude_engine.platform.measurement import clock_ns, exclusive_measurement
from performance.metrics import Record, Sample, TimingPass, Validation
from performance.profiling import HostProfile


class Policy(Record):
    warmup: int = Field(default=3, ge=1)
    repetitions: int = Field(default=20, ge=3)
    host_profile: bool = False


class PreparedBenchmark[M: Record](Protocol):
    @property
    def observation_passes(self) -> tuple[TimingPass, ...]: ...
    def reset(self) -> None: ...
    def invoke(self, *, timing: bool) -> Ticket: ...
    def validate(self, ticket: Ticket) -> Validation: ...
    def assess(self, samples: tuple[Sample, ...]) -> M: ...
    def close(self) -> None: ...


class Procedure[C, W, M: Record](Protocol):
    def prepare(self, component: C, workload: W) -> PreparedBenchmark[M]: ...


class Result[M: Record](Record):
    samples: tuple[Sample, ...]
    validation: Validation
    metrics: M
    host_profile: HostProfile | None = None


def collect[C, W, M: Record](
    procedure: Procedure[C, W, M], component: C, workload: W, policy: Policy
) -> Result[M]:
    with exclusive_measurement():
        case = procedure.prepare(component, workload)
        try:
            case.reset()
            warmup = case.invoke(timing=False)
            warmup.wait()
            for _ in range(policy.warmup - 1):
                case.reset()
                warmup = case.invoke(timing=False)
                warmup.wait()
            validation = case.validate(warmup)
            samples: list[Sample] = []
            host_profile = None
            if validation.passed:
                ticket = warmup
                for pass_kind in case.observation_passes:
                    for _ in range(policy.repetitions):
                        case.reset()
                        started = clock_ns()
                        ticket = case.invoke(timing=pass_kind == TimingPass.DEVICE_EVENTS)
                        ticket.wait()
                        ended = clock_ns()
                        samples.append(
                            Sample(
                                pass_kind=pass_kind,
                                completed_seconds=(ended - started) / 1e9,
                                device_seconds=ticket.device_seconds,
                            )
                        )
                validation = case.validate(ticket)
            if validation.passed and policy.host_profile:
                import cProfile

                from performance.profiling import summarize

                case.reset()
                profiler = cProfile.Profile()
                # Completion and correctness stay outside the host call profile.
                # This invocation does not contribute ordinary timing samples.
                ticket = profiler.runcall(case.invoke, timing=False)
                ticket.wait()
                validation = case.validate(ticket)
                host_profile = summarize(profiler)
            return Result(
                samples=tuple(samples),
                validation=validation,
                metrics=case.assess(tuple(samples)),
                host_profile=host_profile,
            )
        finally:
            case.close()
