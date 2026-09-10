"""Host call evidence from a separate instrumented invocation of the common case."""

import cProfile
from types import CodeType
from typing import Literal

from pydantic import Field

from performance.metrics import Record, Seconds, TimingBoundary


class CallSite(Record):
    file: str
    line: int = Field(ge=0)
    function: str


class HostCall(Record):
    site: CallSite
    primitive_calls: int = Field(ge=0)
    total_calls: int = Field(ge=0)
    own_seconds: Seconds
    cumulative_seconds: Seconds


class HostProfile(Record):
    boundary: Literal[TimingBoundary.PREPARE_THROUGH_SUBMISSION] = (
        TimingBoundary.PREPARE_THROUGH_SUBMISSION
    )
    calls: tuple[HostCall, ...]
    # Instrumentation affects these times; they are attribution evidence, not
    # samples to combine with ordinary latency or device-event observations.
    instrumented: Literal[True] = True


def summarize(profile: cProfile.Profile) -> HostProfile:
    calls = []
    for entry in profile.getstats():
        code = entry.code
        site = (
            CallSite(file=code.co_filename, line=code.co_firstlineno, function=code.co_name)
            if isinstance(code, CodeType)
            else CallSite(file="~", line=0, function=code)
        )
        calls.append(
            HostCall(
                site=site,
                primitive_calls=entry.callcount - entry.reccallcount,
                total_calls=entry.callcount,
                own_seconds=entry.inlinetime,
                cumulative_seconds=entry.totaltime,
            )
        )
    return HostProfile(
        calls=tuple(
            sorted(calls, key=lambda call: (call.site.file, call.site.line, call.site.function))
        )
    )
