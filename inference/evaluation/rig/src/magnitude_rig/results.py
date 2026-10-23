"""What a run tells its caller, and when: each job's result, the answer block, or the failure
with its own error lines and the log they came from; or, once the run passes its estimate,
what it knows so far and what to do about it.

A run is carried out by its own detached process (this command again, with `MRIG_RUN` naming
the run), so it outlives the command that started it. That command follows it and returns at
the first of: the run's end, with its answer and exit code; or its estimate, with the run's
report, still running (`OVERRUN`). `mrig wait` follows it again, to a new estimate.
"""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from . import engine, state
from .connection import HostUnavailable
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
    (`follow`); in the run's own process, carry it out."""
    if CHILD in os.environ:
        return carry_out(arguments, request, answer, os.environ[CHILD])
    run_id = engine.new_run_id(request.instrument)
    results = engine.RESULTS / run_id
    results.mkdir(parents=True)
    with open(results / engine.TRANSCRIPT, "w") as transcript:
        child = subprocess.Popen(
            [sys.executable, "-m", "magnitude_rig.cli", *arguments.argv],
            env={**os.environ, CHILD: run_id}, stdin=subprocess.DEVNULL, stdout=transcript,
            stderr=subprocess.STDOUT, start_new_session=True,
        )

    # Stopping this command stops the run it is following.
    def forward(signum, frame):
        child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    return follow(run_id, results, child.pid, None)


def carry_out(arguments, request: engine.Request, answer: Callable[[dict[str, JobResult]], Answer],
              run_id: str) -> int:
    """Run the request to its end and print its answer, or why it has none, then record the exit
    code: 0 when the answer is good, 2 when it is not (a failing test, a regression), 1 when
    there is none."""
    try:
        outcome = engine.execute(request, run_id)
    except engine.RunFailed as failed:
        code = report_failure(failed)
    except HostUnavailable as error:
        print(f"mrig: {error}")
        code = 1
    else:
        code = report_answer(arguments, answer(job_results(outcome, [job.name for job in request.jobs])), outcome)
    sys.stdout.flush()
    path = engine.RESULTS / run_id / engine.CODE
    staged = path.with_name(f".{engine.CODE}.{os.getpid()}")
    staged.write_text(str(code))
    staged.replace(path)
    return code


def follow(run_id: str, results: Path, pid: int | None, wait: float | None) -> int:
    """Print the run's output and return its exit code once it has ended; once it passes its
    estimate (from its start), or with `wait` seconds from now, print its output so far, its
    report and what to do about it, and return `OVERRUN`."""
    deadline = None if wait is None else time.time() + wait
    passed = "its estimate" if wait is None else f"this wait's estimate ({engine.duration(wait)})"
    while True:
        if deadline is None and (live := state.run_state(run_id)) is not None:
            deadline = live["started"] + live["estimate"]
        if (results / engine.CODE).exists():
            print((results / engine.TRANSCRIPT).read_text(errors="replace"), end="")
            return int((results / engine.CODE).read_text())
        if pid is None or not state.alive(pid):
            print((results / engine.TRANSCRIPT).read_text(errors="replace"), end="")
            print(f"mrig: the process of run {run_id} ended without an answer; its output is above")
            return 1
        if deadline is not None and time.time() >= deadline:
            print((results / engine.TRANSCRIPT).read_text(errors="replace"), end="")
            print(_report(results, pid))
            print(GUIDANCE.format(run=run_id, passed=passed))
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
    print(f"purpose: {arguments.purpose}")
    for line in answer.lines:
        print(line)
    print(
        f"{engine.duration(seconds)} of {engine.duration(arguments.estimate)} estimate · "
        f"{outcome.host} · {outcome.results}"
    )
    return 0 if answer.ok else 2
