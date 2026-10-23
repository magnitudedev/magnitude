"""Running one request on the rig: placement, holds, builds and jobs, phase by phase.

A request names its sides (each a source with its build steps), the device hold its jobs need
(`device`, `timed` or none) and the jobs themselves, in the order they run. On the host it
picks, the run then:

1. holds each side's checkout, evicting least recently used checkouts if a build from cold
   would not fit, and mirrors the side's source into `<root>/checkouts/<key>/src`;
2. under a build hold: installs the recording rustc wrapper, runs the build steps with the
   side's own target, and writes the build provenance of every cargo-built executable the
   side's jobs name (a job never runs an executable its sources did not produce);
3. under the models hold: acquires the request's model into the host's model store, or, for a
   run that needs only its headers, exports the catalog's headers (each component's exact
   header, sparse at the file's full size, so it opens as the model with zero weights);
4. under the device hold (after the thermal gate when timed): runs each job, recorded with the
   host's facts and thermal trace, into `<root>/runs/<run id>/<job>/`;
5. fetches the run directory into `inference/evaluation/results/<run id>/`.

Holds are taken in that order and build and models are released before the device hold is
requested, so runs never wait on each other in a cycle (`host/agent.py`).

Step output goes to `results/<run id>/logs/<phase>.log`, never to the caller. The caller gets
the launch line and, from the instrument, the answer. Every phase's time is kept in
`rig.json` beside the results, with every `mrig wait` on the run and its justification. On
SIGUSR1 the run writes what it knows now (its phases, what it waits behind, its last output,
the device's state) to `report.txt` in its results: what its caller shows when the run passes
its estimate (`results.follow`).

Tokens `{src}`, `{target}`, `{venv}`, `{checkout}` (the side's), `{root}`, `{models}`,
`{python}`, `{artifacts}`, `{catalog}` (the catalog's header export), `{runs}` (the run's
directory, holding every job's), `{inputs}` (the request's files), `{run_dir}` and
`{build_file}` (the job's) are replaced with host paths in build steps, job steps and their
environments.
"""

import getpass
import json
import os
import shlex
import signal
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import state, sync
from .connection import AgentSession, HostUnavailable, Remote, _ssh, connect, host_status
from .host.gate import GateSettings
from .hosts import Hosts
from .sources import Source

RESULTS = sync.INFERENCE / "evaluation" / "results"
# In each run's results: what the run's process printed, its exit code once it has one, its
# latest report, and every wait on it.
TRANSCRIPT, CODE, REPORT, WAITS = "mrig.out", "mrig.code", "report.txt", "waits.jsonl"
HOST_PACKAGE = "evaluation/rig/src"
CATALOG_TOOL = "{target}/release/magnitude-service-catalog-tool"
COMPILER = "RUSTC_WORKSPACE_WRAPPER"  # cargo runs rustc for workspace crates through it
GATE = GateSettings()
NOTE = "rig: "  # how the rig's own lines start in a phase's log
MAKE = "import os, sys; os.makedirs(sys.argv[1], exist_ok=True)"
WRITE = "import os, sys; p = sys.argv[1]; os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'w').write(sys.stdin.read())"


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


CATALOG_BUILD = Step("catalog tool", ("cargo", "build", "--release", "-p", "magnitude-service-catalog-tool"))


@dataclass(frozen=True)
class Side:
    label: str
    source: Source
    builds: tuple[Step, ...]


@dataclass(frozen=True)
class Job:
    side: str  # the label of the side whose build it runs
    name: str  # its run directory under the run's
    step: Step


@dataclass(frozen=True)
class Weights:
    """A catalog model's files, acquired (downloaded only where missing) as `{artifacts}`."""

    model: str


@dataclass(frozen=True)
class Headers:
    """The catalog's exact headers, exported to `{catalog}` (its header bundle and `headers/`),
    and, for a model, its acquisition object over them as `{artifacts}`: the model opens with
    zero weights, nothing of them downloaded."""

    model: str | None  # None: the catalog's headers alone


@dataclass(frozen=True)
class Request:
    instrument: str
    purpose: str
    estimate: float  # seconds, the caller's
    backend: str
    sides: tuple[Side, ...]
    jobs: tuple[Job, ...] = ()
    device: str | None = None  # "device" or "timed" for the jobs; None runs them under build
    host: str | None = None  # a host or pool; None places by backend among every ready host
    acquires: Weights | Headers | None = None  # what the jobs read of the catalog
    files: tuple[Path, ...] = ()  # local files the jobs read, as `{inputs}/<name>`


@dataclass
class Outcome:
    run_id: str
    host: str
    results: Path
    codes: dict[str, int]  # job name -> exit code
    phases: list[dict] = field(default_factory=list)
    stopped: str | None = None


class RunFailed(Exception):
    def __init__(self, phase: str, detail: str, outcome: Outcome):
        super().__init__(f"{phase}: {detail}")
        self.phase, self.detail, self.outcome = phase, detail, outcome


def new_run_id(instrument: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{instrument}-{os.urandom(3).hex()}"


def who() -> str:
    """The holder as other sessions see it: the coding-agent session, the checkout, the user."""
    session = state.agent_session()
    names = [f"session:{session}"] if session else []
    return " ".join([*names, sync.INFERENCE.parent.name, f"{getpass.getuser()}@{socket.gethostname()}"])


def duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


# Placement ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    remote: Remote
    status: dict


def _candidates(request: Request, hosts: Hosts) -> tuple[list[Candidate], list[str]]:
    names = hosts.resolve(request.host) if request.host else tuple(hosts.host(a) for a in hosts.known())
    skipped: list[str] = []

    def probe(config):
        try:
            remote = connect(config)
        except HostUnavailable as error:
            return None, f"{config.alias}: unreachable ({error})"
        if request.backend not in remote.backends:
            return None, None
        if not remote.ready:
            return None, f"{config.alias}: not set up (`mrig setup {config.alias}`)"
        try:
            return Candidate(remote, host_status(remote, request.sides[0].source.key)), None
        except HostUnavailable as error:
            return None, f"{config.alias}: {error}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(probe, names))
    candidates = [c for c, _ in found if c is not None]
    skipped = [reason for _, reason in found if reason]
    return candidates, skipped


def _wait(candidate: Candidate, kind: str) -> float:
    status = candidate.status
    if kind != "build" and status["device_lock_held_elsewhere"] is not None:
        return float("inf")
    return status["waits"][kind]


def place(request: Request, hosts: Hosts) -> Candidate:
    """The candidate that frees earliest for the request's device kind, by the holders' own
    estimates; among equals, one where the checkout is already present."""
    candidates, skipped = _candidates(request, hosts)
    if not candidates:
        where = request.host or "any host"
        detail = "; ".join(skipped) if skipped else f"none runs {request.backend}"
        raise HostUnavailable(f"no {request.backend} host among {where}: {detail}")
    kind = request.device or "build"
    order = {c.remote.alias: i for i, c in enumerate(candidates)}
    return min(
        candidates,
        key=lambda c: (_wait(c, kind), not c.status["checkout"]["present"], order[c.remote.alias]),
    )


def _holders(status: dict) -> str:
    entries = [e for e in status["entries"] if e["state"] == "held" and not e["kind"].startswith("checkout:")]
    if status["device_lock_held_elsewhere"] is not None:
        legacy = status["device_lock_held_elsewhere"]
        entries.append({"kind": "device", "who": legacy.get("who", "another tool"), "run": legacy.get("run")})
    if not entries:
        return "holders: none"
    shown = []
    for entry in entries:
        left = entry.get("remaining")
        timing = f", {duration(left)} left of its estimate" if left is not None else ""
        purpose = f" \"{entry['purpose']}\"" if entry.get("purpose") else ""
        shown.append(f"{entry['kind']} {entry.get('run')}{purpose}{timing}")
    return "holders: " + "; ".join(shown)


def launch_line(request: Request, candidate: Candidate) -> str:
    status = candidate.status
    target = "present" if status["checkout"]["present"] else "none for this checkout (cold build)"
    return (
        f"mrig {request.instrument} · {request.backend} → {candidate.remote.alias} · "
        f"estimate {duration(request.estimate)} · {_holders(status)} · target: {target}"
    )


# Execution ---------------------------------------------------------------------------------


class Logs:
    """Each phase's step output in `logs/<phase>.log`, and the last line any step printed."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.last: tuple[float, str] | None = None
        self._lock = threading.Lock()

    def sink(self, phase: str) -> Callable[[str, str], None]:
        path = self.directory / f"{phase}.log"

        def receive(stream: str, text: str) -> None:
            with self._lock, open(path, "a") as log:
                log.write(text if stream == "stdout" else f"[stderr] {text}")
            if text.strip() and not text.startswith(NOTE):
                self.last = (time.monotonic(), text.strip())

        return receive

    def note(self, phase: str, message: str) -> None:
        self.sink(phase)("stdout", f"{NOTE}{message[:400]}\n")


class Run:
    def __init__(self, request: Request, candidate: Candidate, run_id: str, results: Path):
        self.request = request
        self.remote = candidate.remote
        self.run_id = run_id
        self.results = results
        self.logs = Logs(results / "logs")
        self.started = time.monotonic()
        self.phases: list[dict] = []
        self.phase: str | None = None
        self.phase_started = self.started
        self.state = state.RunState(
            run_id,
            {
                "instrument": request.instrument,
                "purpose": request.purpose,
                "estimate": request.estimate,
                "host": self.remote.alias,
                "results": str(results),
                "started": time.time(),
                "phase": None,
            },
        )
        self.session: AgentSession | None = None
        self.synced = False  # the first side's source, with the host package, is on the host

    # phases

    def enter(self, phase: str) -> None:
        now = time.monotonic()
        if self.phase is not None:
            self.phases.append({"phase": self.phase, "seconds": round(now - self.phase_started, 3)})
        self.phase, self.phase_started = phase, now
        self.state.update(phase=phase, phase_started=time.time())

    def finish(self) -> None:
        self.enter("done")
        self.state.remove()

    # host paths

    def checkout(self, side: Side) -> str:
        return self.remote.path("checkouts", side.source.key)

    def tokens(self, side: Side, job: Job | None) -> dict[str, str]:
        checkout = self.checkout(side)
        run_root = self.remote.path("runs", self.run_id)
        tokens = {
            "checkout": checkout,
            "src": f"{checkout}/src",
            "target": f"{checkout}/target",
            "venv": f"{checkout}/venv",
            "root": self.remote.root,
            "models": self.remote.models,
            "python": self.remote.python,
            "artifacts": f"{run_root}/artifacts.json",
            "catalog": self.remote.path("catalog-headers"),
            "runs": run_root,
            "inputs": f"{run_root}/inputs",
        }
        if job is not None:
            tokens["run_dir"] = f"{run_root}/{job.name}"
            tokens["build_file"] = f"{run_root}/{job.name}/build.json"
        return tokens

    def environment(self, side: Side, step: Step, tokens: dict[str, str]) -> dict[str, str]:
        return {
            "CARGO_TARGET_DIR": tokens["target"],
            "UV_CACHE_DIR": f"{self.remote.root}/cache/uv",
            "UV_PROJECT_ENVIRONMENT": tokens["venv"],
            "PIP_CACHE_DIR": f"{self.remote.root}/cache/pip",
            "HF_HUB_CACHE": f"{self.remote.root}/cache/huggingface/hub",
            COMPILER: f"{tokens['checkout']}/provenance/rustc",
            **self.remote.env,
            **{name: _fill(value, tokens) for name, value in step.env},
        }

    def host_module(self, side: Side, *arguments: str) -> tuple[list[str], dict[str, str]]:
        src = f"{self.checkout(side)}/src"
        env = {"PYTHONPATH": f"{src}/{HOST_PACKAGE}", "PYTHONDONTWRITEBYTECODE": "1"}
        return [self.remote.python, "-m", "magnitude_rig.host", *arguments], env

    # steps

    def execute(self, side: Side, argv: list[str], env: dict[str, str], *, input: str | None = None,
                housekeeping: bool = False) -> tuple[int, str]:
        """Run a step from the side's source root, or, for host housekeeping (which may precede
        that root), from the host's cache root."""
        captured: list[str] = []
        sink = self.logs.sink(self.phase)

        def receive(stream: str, text: str) -> None:
            if stream == "stdout":
                captured.append(text)
            sink(stream, text)

        self.logs.note(self.phase, f"$ {shlex.join(argv)}")
        cwd = self.remote.root if housekeeping else f"{self.checkout(side)}/src"
        code = self.session.run(argv, cwd=cwd, env=env, input=input, on_output=receive)
        return code, "".join(captured)

    def step(self, side: Side, argv: list[str], env: dict[str, str], *, input: str | None = None,
             housekeeping: bool = False) -> str:
        code, stdout = self.execute(side, argv, env, input=input, housekeeping=housekeeping)
        if code != 0:
            raise _StepFailed(self.phase, f"`{shlex.join(argv)[:160]}` exited {code}")
        return stdout

    def make(self, side: Side, path: str) -> None:
        self.step(side, [self.remote.python, "-c", MAKE, path], {}, housekeeping=True)

    def write(self, side: Side, path: str, text: str) -> None:
        self.step(side, [self.remote.python, "-c", WRITE, path], {}, input=text, housekeeping=True)

    def hold(self, kind: str) -> None:
        def waiting(blocking: list[dict]) -> None:
            shown = ", ".join(f"{b['kind']} {b.get('run')}" for b in blocking)
            self.logs.note(self.phase, f"waiting for {kind} behind {shown}")
            self.state.update(waiting_behind=shown)

        self.session.hold(kind, waiting)
        self.state.update(waiting_behind=None)

    # the report

    def write_report(self) -> None:
        """Write what the run knows now to `report.txt`, atomically."""
        path = self.results / REPORT
        staged = path.with_name(f".{REPORT}.{os.getpid()}")
        staged.write_text(self.report() + "\n")
        staged.replace(path)

    def report(self) -> str:
        now = time.monotonic()
        phases = [*self.phases, {"phase": self.phase, "seconds": now - self.phase_started}]
        lines = [
            (f"mrig: {self.request.instrument} {self.run_id} ({self.remote.alias}) at "
             f"{duration(now - self.started)}, estimate {duration(self.request.estimate)}"),
            "  phases:       " + " · ".join(f"{p['phase']} {duration(p['seconds'])}" for p in phases),
        ]
        waiting = self.state.value.get("waiting_behind")
        if waiting:
            lines.append(f"  waiting:      behind {waiting}")
        if self.logs.last is not None:
            at, text = self.logs.last
            lines.append(f"  last output:  {duration(now - at)} ago, {text[:160]!r}")
        else:
            lines.append("  last output:  none yet")
        lines.append(f"  device:       {self.device_facts()}")
        lines.append(f"  log:          {self.logs.directory / (self.phase + '.log')}")
        lines.append(f"  stop:         mrig stop {self.run_id}")
        return "\n".join(lines)

    def device_facts(self) -> str:
        if not self.synced:
            return "not read yet: the host's probe arrives with the run's source, still syncing"
        side = self.request.sides[0]
        argv, env = self.host_module(side, "probe")
        line = shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *argv])
        try:
            reading = json.loads(_ssh(self.remote, line, timeout=30))
        except (HostUnavailable, ValueError) as error:
            return f"unavailable ({error})"
        utilization = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(reading["utilization_pct"].items()))
        return utilization or f"no utilization reading ({reading['errors']})"


class _StepFailed(Exception):
    def __init__(self, phase: str, detail: str):
        super().__init__(detail)
        self.phase = phase


def _fill(text: str, tokens: dict[str, str]) -> str:
    for token, value in tokens.items():
        text = text.replace("{" + token + "}", value)
    return text


def _substitute(argv: tuple[str, ...], tokens: dict[str, str]) -> list[str]:
    return [_fill(argument, tokens) for argument in argv]


def _texts(run: Run, side: Side) -> list[str]:
    """The side's job texts with host paths, which name the executables to prove."""
    texts = []
    for job in run.request.jobs:
        if job.side == side.label:
            tokens = run.tokens(side, job)
            texts += _substitute(job.step.argv, tokens)
            texts += [_fill(value, tokens) for _, value in job.step.env]
    return texts


def _prepare(run: Run, side: Side) -> None:
    """Checkout hold, eviction and mirror of one side's source. The checkout hold, which every
    run of this tree shares, keeps it from eviction for the whole run; the mirror hold, which
    one run at a time takes, covers only the mirror and prune."""
    session, remote = run.session, run.remote
    checkout = run.checkout(side)
    run.hold(f"checkout:{side.source.key}")
    run.enter(f"queue-mirror-{side.label}")
    run.hold(f"mirror:{side.source.key}")
    run.enter(f"sync-{side.label}")
    evicted = session.evict(side.source.key)
    if evicted["removed"]:
        run.logs.note(run.phase, f"evicted least recently used checkouts {evicted['removed']}")
    started = time.monotonic()
    run.make(side, f"{checkout}/src")
    sync.push(remote, list(side.source.paths), side.source.directory, f"{checkout}/src")
    run.step(
        side,
        [remote.python, remote.agent, "prune", "--dir", f"{checkout}/src"],
        {},
        input="\n".join(side.source.paths) + "\n",
    )
    session.release(f"mirror:{side.source.key}")
    run.logs.note(run.phase, f"mirrored {len(side.source.paths)} files in {time.monotonic() - started:.1f} s")


def _build(run: Run, side: Side, cold: bool) -> None:
    run.enter(f"build-{side.label}")
    argv, env = run.host_module(side, "compiler", "--root", run.checkout(side))
    run.step(side, argv, env)
    request = run.request
    acquires = side is request.sides[0] and request.acquires is not None
    for step in (*side.builds, *((CATALOG_BUILD,) if acquires else ())):
        tokens = run.tokens(side, None)
        run.step(side, _substitute(step.argv, tokens), run.environment(side, step, tokens))
    texts = _texts(run, side)
    if texts:
        tokens = run.tokens(side, None)
        output = run.remote.path("runs", run.run_id, f"provenance-{side.label}.json")
        argv, env = run.host_module(
            side, "provenance", "--output", output, "--source", tokens["src"], "--", *texts
        )
        run.logs.note(run.phase, run.step(side, argv, env).strip())
    if cold:
        argv, env = run.host_module(side, "size", "--checkout", run.checkout(side))
        run.step(side, argv, env)


def _acquire(run: Run, side: Side) -> None:
    run.enter("queue-models")
    run.hold("models")
    run.enter("models")
    tokens = run.tokens(side, None)
    env = run.environment(side, Step("models", ()), tokens)
    tool = _fill(CATALOG_TOOL, tokens)
    acquires = run.request.acquires
    if isinstance(acquires, Headers):
        exported = run.step(
            side, [tool, "headers", "--output", tokens["catalog"], "--model-store", run.remote.models], env
        ).strip().splitlines()[-1]
        if acquires.model is not None:
            argv, module_env = run.host_module(
                side, "header-artifacts", "--export", exported, "--model", acquires.model,
                "--output", tokens["artifacts"],
            )
            run.step(side, argv, module_env)
    else:
        output = run.step(
            side, [tool, "acquire", "--model", acquires.model, "--model-store", run.remote.models], env
        )
        run.write(side, tokens["artifacts"], json.dumps(json.loads(output), indent=2))
    run.session.release("models")


def _jobs(run: Run, codes: dict[str, int]) -> None:
    request = run.request
    sides = {side.label: side for side in request.sides}
    if request.device is not None:
        run.enter(f"queue-{request.device}")
        run.hold(request.device)
    elif request.jobs:  # host work that needs no device still never runs during a measurement
        run.enter("queue-build")
        run.hold("build")
    if request.device == "timed":
        run.enter("gate")
        side = request.sides[0]
        settings = [
            argument
            for name, value in vars(GATE).items()
            for argument in (f"--{name.replace('_', '-')}", str(value))
        ]
        argv, env = run.host_module(side, "gate", "--log", run.remote.path("runs", run.run_id, "thermal-gate.jsonl"), *settings)
        run.step(side, argv, env)
    for job in request.jobs:
        side = sides[job.side]
        run.enter(f"job-{job.name}")
        tokens = run.tokens(side, job)
        run.write(side, tokens["build_file"], json.dumps(side.source.identity.to_json(), indent=2))
        record, record_env = run.host_module(
            side, "record", "--run-dir", tokens["run_dir"], "--alias", run.remote.alias, "--",
            *_substitute(job.step.argv, tokens),
        )
        env = {**run.environment(side, job.step, tokens), **record_env}
        codes[job.name], _ = run.execute(side, record, env)


def _fetch(run: Run) -> None:
    run.enter("fetch")
    run_dir = run.remote.path("runs", run.run_id)
    exists = [run.remote.python, "-c", "import os, sys; sys.exit(not os.path.isdir(sys.argv[1]))", run_dir]
    code, _ = run.execute(run.request.sides[0], exists, {}, housekeeping=True)
    if code == 0:
        sync.fetch(run.remote, run_dir, run.results)
        remove = [run.remote.python, "-c", "import shutil, sys; shutil.rmtree(sys.argv[1])", run_dir]
        run.execute(run.request.sides[0], remove, {}, housekeeping=True)


def _salvage(run: Run) -> str:
    """Fetch what a failed or stopped run left on the host; what keeps it there, if anything."""
    if run.session is None:
        return ""
    try:
        _fetch(run)
    except (_StepFailed, HostUnavailable, RuntimeError, OSError) as error:
        return f"; its host run directory was not fetched: {error}"
    return ""


def _announce(line: str) -> None:
    print(line, flush=True)


def execute(request: Request, run_id: str, *, hosts: Hosts | None = None,
            announce: Callable[[str], None] = _announce) -> Outcome:
    """Run `request` as `run_id` to its end and return where its results are; raises RunFailed
    when a phase before the jobs fails (the jobs' own exit codes are the outcome's). SIGUSR1
    writes the run's report meanwhile."""
    results = RESULTS / run_id
    current: list[Run] = []

    def reported(signum, frame):
        threading.Thread(target=_write_report, args=(current, request, results), daemon=True).start()

    previous = signal.signal(signal.SIGUSR1, reported)
    try:
        return _execute(request, run_id, results, hosts or Hosts.load(), announce, current)
    finally:
        signal.signal(signal.SIGUSR1, previous)


def _write_report(current: list[Run], request: Request, results: Path) -> None:
    if current:
        current[0].write_report()
        return
    results.mkdir(parents=True, exist_ok=True)
    path = results / REPORT
    staged = path.with_name(f".{REPORT}.{os.getpid()}")
    staged.write_text(f"mrig: {request.instrument} is still choosing its host\n")
    staged.replace(path)


def _execute(request: Request, run_id: str, results: Path, hosts: Hosts,
             announce: Callable[[str], None], current: list[Run]) -> Outcome:
    candidate = place(request, hosts)
    announce(launch_line(request, candidate))
    run = Run(request, candidate, run_id, results)
    current.append(run)
    outcome = Outcome(run_id, candidate.remote.alias, results, {})
    stop = threading.Event()

    def stopped(signum, frame):
        stop.set()
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stopped)
    run.enter("connect")
    try:
        run.session = AgentSession(
            run.remote, who=who(), run=run_id, purpose=request.purpose, estimate=request.estimate
        )
        run.make(request.sides[0], run.remote.path("runs", run_id))
        presence = {}
        for side in request.sides:
            _prepare(run, side)
            run.synced = True
            presence[side.label] = host_status(run.remote, side.source.key)["checkout"]["present"]
        if request.files:
            inputs = run.tokens(request.sides[0], None)["inputs"]
            run.make(request.sides[0], inputs)
            for path in request.files:
                sync.push(run.remote, [path.name], path.parent, inputs)
        run.enter("queue-build")
        run.hold("build")
        for side in request.sides:
            _build(run, side, cold=not presence[side.label])
        run.session.release("build")
        if request.acquires is not None:
            _acquire(run, request.sides[0])
        _jobs(run, outcome.codes)
        _fetch(run)
    except (_StepFailed, KeyboardInterrupt, HostUnavailable, RuntimeError) as error:
        phase = error.phase if isinstance(error, _StepFailed) else run.phase
        if isinstance(error, KeyboardInterrupt):
            outcome.stopped = "stopped by mrig stop" if stop.is_set() else "interrupted"
            if run.session is not None:
                run.session.kill()
        detail = outcome.stopped or str(error)
        detail += _salvage(run)
        _record(run, outcome, failed=(phase, detail))
        raise RunFailed(phase, detail, outcome) from None
    finally:
        signal.signal(signal.SIGTERM, previous)
        if run.session is not None:
            run.session.close()
    _record(run, outcome, failed=None)
    return outcome


def waits(results: Path) -> list[dict]:
    """Every `mrig wait` on the run: when, the new estimate and why it was needed."""
    path = results / WAITS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _record(run: Run, outcome: Outcome, failed: tuple[str, str] | None) -> None:
    run.finish()
    outcome.phases = run.phases
    request = run.request
    record = {
        "run": run.run_id,
        "instrument": request.instrument,
        "purpose": request.purpose,
        "estimate_seconds": request.estimate,
        "seconds": round(sum(p["seconds"] for p in run.phases), 3),
        "phases": run.phases,
        "host": run.remote.alias,
        "backend": request.backend,
        "sources": {
            side.label: {"name": side.source.name, "key": side.source.key, **side.source.identity.to_json()}
            for side in request.sides
        },
        "acquires": None if request.acquires is None else {
            "kind": type(request.acquires).__name__.lower(), "model": request.acquires.model,
        },
        "codes": outcome.codes,
        "failed": None if failed is None else {"phase": failed[0], "detail": failed[1]},
        "waits": waits(run.results),
    }
    run.results.mkdir(parents=True, exist_ok=True)
    (run.results / "rig.json").write_text(json.dumps(record, indent=2))
