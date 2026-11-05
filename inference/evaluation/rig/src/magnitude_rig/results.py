"""What a run tells its caller, and when: each job's result, the answer block, or the failure
with its own error lines and the log they came from; or, once the run passes its estimate,
what it knows so far and what to do about it.

A run is carried out by its own detached process (this command again, with `MRIG_RUN` naming
the run), so it outlives the command that started it. That command follows it and returns at
the first of: the run's end, with its answer and exit code; or its estimate, with the run's
report, still running (`OVERRUN`). `mrig wait` follows it again, to a new estimate.
"""

import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from . import engine, state, store
from .connection import HostError
from .instruments import Answer, JobResult
from .instruments.cargo import failures

CHILD = "MRIG_RUN"  # set in a run's own process: the run id it carries out
OVERRUN = 3  # the exit code of a command that returned at the estimate, the run still going
POLL = 0.5  # seconds between looks at the run's results while following it
REPORT_SECONDS = 45  # how long a report may take to write (it reads the device over ssh)
GUIDANCE = """\
The run passed {passed} and is still running. Before anything else, work out why: the
phases above show where the time went, and a run that takes longer than you expected means your
picture of the system is wrong. Then ask whether a faster check would decide the same question
(AGENTS.md, "Running checks and measurements"); if one would, stop this run and run that:
  mrig stop {run}
Wait only if this duration is truly necessary and the estimate was wrong for a reason you can
name:
  mrig wait {run} --estimate DURATION --justification "why it needs DURATION\""""


def job_results(outcome: engine.Outcome, names: list[str]) -> dict[str, JobResult]:
    """Each job's result by name; with no jobs, the build's (its log) as "build"."""
    if not names:
        return {"build": JobResult(outcome.results, outcome.results / "logs" / "build-source.log", 0)}
    return {
        name: JobResult(outcome.results / name, outcome.results / "logs" / f"job-{name}.log", outcome.codes[name])
        for name in names
    }


def perform(arguments, request: engine.Request, answer: Callable[[dict[str, JobResult]], Answer]) -> int:
    """Start the request's run in its own process and follow it to its end or its estimate
    (`follow`); in the run's own process, carry it out. (A run fetched without its caller is
    answered from its fetched results alone, `answer_fetched`, before any source is opened.)"""
    if CHILD in os.environ:
        return carry_out(arguments, request, answer, os.environ[CHILD])
    run_id, results, child = launch(request.instrument, arguments.argv)
    forward_stops([child])
    return follow(run_id, results, child.pid, None)


def launch(instrument: str, argv: list[str]) -> tuple[str, Path, subprocess.Popen]:
    """Start a run of `instrument` in its own detached process: the mrig command `argv` again,
    carrying out the run its `CHILD` names (`perform`). Its id, its results and its process."""
    run_id = engine.new_run_id(instrument)
    results = store.results() / run_id
    results.mkdir(parents=True)
    with open(results / engine.TRANSCRIPT, "w") as transcript:
        child = subprocess.Popen(
            [sys.executable, "-m", "magnitude_rig.cli", *argv],
            env={**os.environ, CHILD: run_id}, stdin=subprocess.DEVNULL, stdout=transcript,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    return run_id, results, child


def forward_stops(children: list[subprocess.Popen]) -> None:
    """Stopping this command stops the runs it is following."""

    def forward(signum, frame):
        for child in children:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)


def carry_out(arguments, request: engine.Request, answer: Callable[[dict[str, JobResult]], Answer],
              run_id: str) -> int:
    """Run the request to its end and print its answer, or why it has none, then record the exit
    code: 0 when the answer is good, 2 when it is not (a failing test, a regression), 1 when
    there is none. The code ends the command following it (`follow`); this process then fetches
    the rest of an ended run (`engine.complete`). A run lost to an error here is left to the
    next command (`engine.collect`)."""
    outcome = None
    try:
        outcome = engine.execute(request, run_id, command=tuple(arguments.argv))
    except engine.RunFailed as failed:
        outcome, code = failed.outcome, report_failure(failed)
    except HostError as error:
        print(f"mrig: {error}")
        code = 1
    else:
        code = report_answer(arguments, answer(job_results(outcome, list(outcome.codes))), outcome)
    sys.stdout.flush()
    _write_code(store.results() / run_id, code)
    if outcome is not None:
        engine.complete(run_id)
    return code


def _write_code(results: Path, code: int) -> None:
    path = results / engine.CODE
    staged = path.with_name(f".{engine.CODE}.{os.getpid()}")
    staged.write_text(str(code))
    staged.replace(path)


def answer_fetched(arguments, answer: Callable[[dict[str, JobResult]], Answer], run_id: str) -> int:
    """The answer of a run fetched without its caller, from its results alone (its `rig.json`
    and its jobs' directories and logs), printed as its caller would have."""
    results = store.results() / run_id
    record = json.loads((results / "rig.json").read_text())
    outcome = engine.Outcome(run_id, record["host"], results, record["codes"], record["phases"],
                             record.get("estimate_seconds"))
    if record["failed"] is not None:
        return report_failure(engine.RunFailed(record["failed"]["phase"], record["failed"]["detail"], outcome))
    return report_answer(arguments, answer(job_results(outcome, list(record["codes"]))), outcome)


def record_answer(results: Path, text: str, code: int) -> None:
    """Append a collected run's answer to its transcript and record its exit code, as its
    caller would have."""
    with open(results / engine.TRANSCRIPT, "a") as transcript:
        transcript.write(text)
    _write_code(results, code)


def follow(run_id: str, results: Path, pid: int | None, wait: float | None, out: TextIO | None = None) -> int:
    """Print the run's output to `out` (this command's standard output by default) and return
    its exit code once it has ended; once its held time passes its estimate, or `wait` more
    seconds of held time from now (`state.account`: time queued for a hold never counts), print
    its output so far, its report and what to do about it, and return `OVERRUN`. A `wait` of
    `math.inf` follows it to its end."""
    out = sys.stdout if out is None else out
    limit = None  # the held seconds at which it is overrun, once the run is live
    if wait is None:
        passed = "its estimate"
    elif math.isfinite(wait):
        passed = f"this wait's estimate ({engine.duration(wait)})"
    while True:
        held = None
        if (live := state.run_state(run_id)) is not None:
            held = state.account(live, time.time())[1]
            if limit is None:
                limit = live["estimate"] if wait is None else held + wait
        if (results / engine.CODE).exists():
            print((results / engine.TRANSCRIPT).read_text(errors="replace"), end="", file=out)
            return int((results / engine.CODE).read_text())
        if pid is None or not state.alive(pid):
            print((results / engine.TRANSCRIPT).read_text(errors="replace"), end="", file=out)
            print(f"mrig: the process of run {run_id} ended without an answer; its output is above", file=out)
            return 1
        if held is not None and held >= limit:
            print((results / engine.TRANSCRIPT).read_text(errors="replace"), end="", file=out)
            print(_report(results, pid), file=out)
            print(GUIDANCE.format(run=run_id, passed=passed), file=out)
            return OVERRUN
        time.sleep(POLL)


def _report(results: Path, pid: int) -> str:
    """The run's report as its process writes it on request."""
    path = results / engine.REPORT
    path.unlink(missing_ok=True)  # the report that appears next is the one asked for
    asked = time.time()
    os.kill(pid, signal.SIGUSR1)
    while time.time() < asked + REPORT_SECONDS:
        if path.exists():
            return path.read_text().rstrip()
        time.sleep(POLL)
    return f"mrig: the run did not write its report within {REPORT_SECONDS} s"


def report_failure(failed: engine.RunFailed) -> int:
    log = failed.outcome.results / "logs" / f"{failed.phase}.log"
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    shown = failures(lines) or [line for line in lines if not line.startswith(engine.NOTE)][-20:]
    print(f"FAILED in {failed.phase}: {failed.detail}")
    for line in shown:
        print(f"  {line}")
    print(f"log: {log}")
    return 1


def report_answer(arguments, answer: Answer, outcome: engine.Outcome) -> int:
    seconds = sum(phase["seconds"] for phase in outcome.phases)
    # As placed (a derived estimate is known only then), else as asked; a run its caller did
    # not record may have neither.
    estimate = outcome.estimate if outcome.estimate is not None else arguments.estimate
    against = "an unrecorded" if estimate is None else engine.duration(estimate)
    print(f"purpose: {arguments.purpose}")
    for line in answer.lines:
        print(line)
    print(f"{engine.duration(seconds)} of {against} estimate · {outcome.host} · {outcome.results}")
    return 0 if answer.ok else 2
