"""Instrumented host calls remain separate from latency and device observations."""

import sys

import pytest

from performance.metrics import Record, TimingBoundary, TimingPass, Validation
from performance.runner import Policy, Result, collect


class Metrics(Record):
    samples: int


class Completion:
    device_seconds = 0.001

    def __init__(self):
        self.completed = False

    def wait(self):
        self.completed = True


class Case:
    observation_passes = (TimingPass.COMPLETED, TimingPass.DEVICE_EVENTS)

    def __init__(self, *, fail_validation=False, fail_profile=False):
        self.invocations = []
        self.resets = 0
        self.validations = 0
        self.closed = False
        self.fail_validation, self.fail_profile = fail_validation, fail_profile

    def reset(self):
        self.resets += 1

    def invoke(self, *, timing):
        self.invocations.append(timing)
        if self.fail_profile and len(self.invocations) == 9:
            raise ValueError("profiled preparation failed")
        return Completion()

    def validate(self, ticket):
        assert ticket.completed
        self.validations += 1
        return Validation(
            passed=not self.fail_validation,
            method="fixture",
            maximum_absolute_error=None,
            maximum_bound_fraction=None,
            failure="bad output" if self.fail_validation else None,
        )

    def assess(self, samples):
        return Metrics(samples=len(samples))

    def close(self):
        self.closed = True


class Procedure:
    def prepare(self, component, workload):
        return component


def test_profile_uses_one_separate_reset_invocation_and_validates_its_output():
    case = Case()
    result = collect(Procedure(), case, None, Policy(warmup=2, repetitions=3, host_profile=True))
    assert case.invocations == [False] * 5 + [True] * 3 + [False]
    assert case.resets == 9 and case.validations == 3 and case.closed
    assert len(result.samples) == 6 and result.metrics.samples == 6
    profile = result.host_profile
    assert profile is not None and profile.instrumented
    assert profile.boundary == TimingBoundary.PREPARE_THROUGH_SUBMISSION
    calls = [c for c in profile.calls if c.site.file == __file__ and c.site.function == "invoke"]
    assert len(calls) == 1 and calls[0].total_calls == 1
    assert not any(c.site.function in ("wait", "reset", "validate") for c in profile.calls)
    assert Result[Metrics].model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("host_profile", [False, True])
def test_failed_qualification_never_profiles_or_collects_latency(host_profile):
    case = Case(fail_validation=True)
    result = collect(
        Procedure(), case, None, Policy(warmup=2, repetitions=3, host_profile=host_profile)
    )
    assert not result.validation.passed and result.host_profile is None
    assert not result.samples and case.invocations == [False, False] and case.closed


def test_profile_failure_restores_interpreter_profiler_and_retires_case():
    case = Case(fail_profile=True)
    previous = sys.getprofile()
    with pytest.raises(ValueError, match="profiled preparation failed"):
        collect(Procedure(), case, None, Policy(warmup=2, repetitions=3, host_profile=True))
    assert sys.getprofile() is previous
    assert case.closed and len(case.invocations) == 9
