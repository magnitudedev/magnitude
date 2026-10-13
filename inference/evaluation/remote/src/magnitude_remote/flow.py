"""Running one evaluation job, on a host under its agent's lock or locally.

Steps, in order, all as children of the lock-holding agent on a host:
sync (remote only) -> build -> build provenance -> catalog acquisition -> thermal gate ->
job (recorded) -> fetch.
The job's run directory is `<runs>/<run id>/` on the executing host and is copied to
`inference/evaluation/results/<run id>/` (local runs write there directly).

Every step compiles workspace crates through the recording rustc wrapper, and after the
build the flow writes `{run_dir}/provenance.json`: for each cargo-built executable the job's
command, build commands or environment name, whether every unit it was linked from was
compiled from the bytes now in the synchronized sources (`host/provenance.py`). The job does
not run unless each is verified, so a run's build identity describes the binary that ran.

Tokens `{run_dir}`, `{artifacts}`, `{src}`, `{target}`, `{models}`, `{python}`, `{root}`
(the host's cache root) and `{tree:<name>}` (each of the job's extra trees) are replaced with
that host's paths in the job command, its build commands and its environment. `{uv}` is the
host's uv as the job PATH resolves it (`~/.local/bin` and `~/.cargo/bin` come first), and
`{build_file}` is `{run_dir}/build.json`, the controller's build identity (contract JSON)
written before the build. Every step runs from the source root.

A job's extra trees are local directories mirrored to `<root>/trees/<name>/` on a host before
the job, like the source tree; a local run uses them in place.
"""

import getpass
import json
import os
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from . import sync
from .build import BuildIdentity, identify
from .connection import AgentSession, HostUnavailable, Remote, connect
from .host.gate import GateSettings
from .hosts import HostConfig, Hosts

RESULTS = sync.INFERENCE / "evaluation" / "results"
HOST_PACKAGE = "evaluation/remote/src"
CATALOG_ACQUIRE = (
    "cargo", "run", "--release", "--quiet", "-p", "magnitude-service-catalog-tool", "--", "acquire"
)
PROBED_TOOLS = ("uv",)  # `{uv}` resolves to the host's own executable, probed on use
COMPILER = "RUSTC_WORKSPACE_WRAPPER"  # cargo runs rustc for workspace crates through it
# The environment variable each coding agent gives its commands to identify its session.
AGENT_SESSIONS = (("claude", "CLAUDE_CODE_SESSION_ID"), ("codex", "CODEX_THREAD_ID"))
OutputSink = Callable[[str, str, str], None]  # (step, stream, text)


@dataclass(frozen=True)
class Tree:
    """A local directory the job needs on the host besides the source tree: `paths` (relative
    to `local`) are mirrored to `<root>/trees/<name>/` by content, like the source tree."""

    name: str
    local: Path
    paths: tuple[str, ...]


@dataclass(frozen=True)
class Job:
    tool: str
    command: tuple[str, ...]
    build: tuple[tuple[str, ...], ...] = ()
    model: str | None = None
    gate: GateSettings | None = field(default_factory=GateSettings)
    max_hold_seconds: float = 6 * 3600
    trees: tuple[Tree, ...] = ()
    env: tuple[tuple[str, str], ...] = ()  # (name, value) for the build and job steps


@dataclass(frozen=True)
class Outcome:
    run_id: str
    host: str | None
    results: Path
    code: int


class StepFailed(Exception):
    def __init__(self, step: str, code: int, detail: str = ""):
        super().__init__(f"{step} exited {code}" + (f": {detail}" if detail else ""))
        self.step, self.code = step, code


@dataclass(frozen=True)
class Layout:
    """Paths on the executing host."""

    src: str
    target: str
    runs: str
    models: str
    python: str
    root: str


class Executor(Protocol):
    alias: str | None
    layout: Layout

    def run(self, step: str, argv: list[str], *, env: dict[str, str], input: str | None) -> tuple[int, str]: ...

    def tree(self, tree: Tree) -> str:
        """Where `tree` is on the executing host."""
        ...


def _collect(step: str, sink: OutputSink, captured: list[str]) -> Callable[[str, str], None]:
    def receive(stream: str, text: str) -> None:
        if stream == "stdout":
            captured.append(text)
        sink(step, stream, text)

    return receive


class LocalExecutor:
    alias = None

    def __init__(self, config: HostConfig, sink: OutputSink):
        source = sync.INFERENCE
        self.layout = Layout(
            str(source),
            str(source / "target"),
            str(RESULTS),
            str(Path(config.models).expanduser()),
            sys.executable,
            str(Path(config.root).expanduser()),
        )
        self._sink = sink
        self._env = {name: os.path.expanduser(value) for name, value in config.env}

    def tree(self, tree):
        return str(tree.local)

    def run(self, step, argv, *, env, input):
        captured: list[str] = []
        receive = _collect(step, self._sink, captured)
        process = subprocess.Popen(
            argv,
            cwd=self.layout.src,
            env={**os.environ, **self._env, **env},
            stdin=subprocess.DEVNULL if input is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        pumps = [
            threading.Thread(target=lambda s=s, n=n: [receive(n, line) for line in s])
            for s, n in ((process.stdout, "stdout"), (process.stderr, "stderr"))
        ]
        for pump in pumps:
            pump.start()
        try:
            if input is not None:
                process.stdin.write(input)
                process.stdin.close()
            code = process.wait()
        except BaseException:
            _kill_group(process)
            raise
        for pump in pumps:
            pump.join()
        return code, "".join(captured)


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def host_caches(layout: Layout) -> dict[str, str]:
    """Every step's build outputs and tool caches live under the host's cache root, never on
    its system disk: cargo targets, uv, pip and the Hugging Face hub cache."""
    return {
        "CARGO_TARGET_DIR": layout.target,
        "UV_CACHE_DIR": f"{layout.root}/cache/uv",
        "PIP_CACHE_DIR": f"{layout.root}/cache/pip",
        "HF_HUB_CACHE": f"{layout.root}/cache/huggingface/hub",
    }


class RemoteExecutor:
    def __init__(self, session: AgentSession, sink: OutputSink):
        remote = session.remote
        self.remote = remote
        self.alias = remote.alias
        self.layout = Layout(
            remote.path("src"),
            remote.path("target"),
            remote.path("runs"),
            remote.models,
            remote.python,
            remote.root,
        )
        self._session = session
        self._sink = sink

    def tree(self, tree):
        return self.remote.path("trees", tree.name)

    def run(self, step, argv, *, env, input):
        captured: list[str] = []
        code = self._session.run(
            argv,
            cwd=self.layout.src,
            env={**host_caches(self.layout), **self.remote.env, **env},
            input=input,
            on_output=_collect(step, self._sink, captured),
        )
        return code, "".join(captured)


def new_run_id(tool: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{tool}-{secrets.token_hex(3)}"


def _who() -> str:
    """The lock holder as other sessions see it: the coding agent session running this, when
    there is one, then the checkout's directory name, then the user and machine."""
    sessions = [f"{agent}:{os.environ[name]}" for agent, name in AGENT_SESSIONS if name in os.environ]
    return " ".join([*sessions, sync.INFERENCE.parent.name, f"{getpass.getuser()}@{socket.gethostname()}"])


def _report_holder(alias: str, log: Callable[[str], None]) -> Callable[[dict | None, int], None]:
    def report(holder: dict | None, ahead: int) -> None:
        held = (
            "lock busy (holder unknown)"
            if holder is None
            else f"held by {holder['who']} for run {holder['run']} since {holder['since']}"
        )
        log(f"{alias}: {held}; waiting with {ahead} queued ahead")

    return report


@contextmanager
def acquire(
    configs: tuple[HostConfig, ...], *, run: str, max_hold: float, log: Callable[[str], None]
) -> Iterator[AgentSession]:
    """Hold the lock of whichever pool member frees first (a single host is a pool of one)."""
    remotes: list[Remote] = []
    for config in configs:
        try:
            remotes.append(connect(config))
        except HostUnavailable as error:
            log(f"skipping {config.alias}: {error}")
    if not remotes:
        raise HostUnavailable(f"no reachable host among {[c.alias for c in configs]}")
    sessions = [AgentSession(r, who=_who(), run=run, max_hold=max_hold) for r in remotes]
    reporters = {s.remote.alias: _report_holder(s.remote.alias, log) for s in sessions}
    winner = None
    try:
        while winner is None:
            if not sessions:
                raise HostUnavailable(f"every agent session ended among {[c.alias for c in configs]}")
            for session in list(sessions):
                try:
                    if session.poll_locked(0.2, reporters[session.remote.alias]):
                        winner = session
                        break
                except HostUnavailable as error:
                    log(str(error))
                    sessions.remove(session)
        for session in sessions:
            if session is not winner:
                session.close()
        log(f"{winner.remote.alias}: lock held ({winner.remote.system} {winner.remote.machine})")
        yield winner
    finally:
        if winner is not None:
            winner.close()
        else:
            for session in sessions:
                session.close()


def _fill(text: str, tokens: dict[str, str]) -> str:
    for token, value in tokens.items():
        text = text.replace("{" + token + "}", value)
    return text


def _substitute(argv: tuple[str, ...], tokens: dict[str, str]) -> list[str]:
    return [_fill(argument, tokens) for argument in argv]


def _host_module(layout: Layout, *arguments: str) -> list[str]:
    return [layout.python, "-m", "magnitude_remote.host", *arguments]


def _host_env(layout: Layout) -> dict[str, str]:
    """Run the synchronized host package without writing bytecode into the mirror."""
    return {"PYTHONPATH": f"{layout.src}/{HOST_PACKAGE}", "PYTHONDONTWRITEBYTECODE": "1"}


def _step(
    executor: Executor,
    step: str,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> str:
    code, stdout = executor.run(step, argv, env=env or {}, input=input)
    if code != 0:
        raise StepFailed(step, code, stdout.strip().splitlines()[-1] if stdout.strip() else "")
    return stdout


def _write(executor: Executor, path: str, text: str) -> None:
    script = (
        "import os, sys; p = sys.argv[1]; os.makedirs(os.path.dirname(p), exist_ok=True); "
        "open(p, 'w').write(sys.stdin.read())"
    )
    _step(executor, "write", [executor.layout.python, "-c", script, path], input=text)


def _which(executor: Executor, tool: str) -> str:
    """The tool's absolute path as the executing host's job environment resolves it."""
    script = "import shutil, sys; p = shutil.which(sys.argv[1]); print(p or ''); sys.exit(p is None)"
    code, stdout = executor.run(
        f"which {tool}", [executor.layout.python, "-c", script, tool], env={}, input=None
    )
    if code != 0:
        raise StepFailed(f"which {tool}", code, f"{tool} is not on the job PATH")
    return stdout.strip()


def perform(
    executor: Executor, job: Job, run_id: str, build: BuildIdentity, log: Callable[[str], None]
) -> int:
    """Build, acquire, gate and run the recorded job; returns the job's exit code."""
    layout = executor.layout
    run_dir = f"{layout.runs}/{run_id}"
    artifacts = f"{run_dir}/artifacts.json"
    tokens = {
        "run_dir": run_dir,
        "artifacts": artifacts,
        "src": layout.src,
        "target": layout.target,
        "models": layout.models,
        "python": layout.python,
        "root": layout.root,
        "build_file": f"{run_dir}/build.json",
        **{f"tree:{tree.name}": executor.tree(tree) for tree in job.trees},
    }
    texts = [*job.command, *(a for argv in job.build for a in argv), *(v for _, v in job.env)]
    for tool in PROBED_TOOLS:
        if any("{" + tool + "}" in text for text in texts):
            tokens[tool] = _which(executor, tool)
    compiler = _step(
        executor, "compiler", _host_module(layout, "compiler", "--root", layout.root), env=_host_env(layout)
    )
    recorded = {COMPILER: compiler.strip()}
    env = {**recorded, **{name: _fill(value, tokens) for name, value in job.env}}
    _write(executor, tokens["build_file"], json.dumps(build.to_json(), indent=2))
    for argv in job.build:
        build = _substitute(argv, tokens)
        log(f"build: {shlex.join(build)}")
        _step(executor, "build", build, env=env)
    sources = (layout.src, *(executor.tree(tree) for tree in job.trees))
    prove = _host_module(
        layout,
        "provenance",
        "--output",
        f"{run_dir}/provenance.json",
        *(argument for source in sources for argument in ("--source", source)),
        "--",
        *(_fill(text, tokens) for text in texts),
    )
    log(f"provenance: {_step(executor, 'provenance', prove, env=_host_env(layout)).strip()}")
    if job.model is not None:
        log(f"acquire: {job.model} into {layout.models}")
        output = _step(
            executor,
            "acquire",
            [*CATALOG_ACQUIRE, "--model", job.model, "--model-store", layout.models],
            env=recorded,
        )
        _write(executor, artifacts, json.dumps(json.loads(output), indent=2))
    if job.gate is not None:
        log("thermal gate: waiting for flat temperatures and an idle host")
        settings = [
            argument
            for f in fields(GateSettings)
            for argument in (f"--{f.name.replace('_', '-')}", str(getattr(job.gate, f.name)))
        ]
        gate = [*_host_module(layout, "gate", "--log", f"{run_dir}/thermal-gate.jsonl"), *settings]
        _step(executor, "gate", gate, env=_host_env(layout))
    alias = ["--alias", executor.alias] if executor.alias else []
    record = _host_module(layout, "record", "--run-dir", run_dir, *alias, "--", *_substitute(job.command, tokens))
    log(f"job: {shlex.join(_substitute(job.command, tokens))}")
    code, _ = executor.run("job", record, env={**env, **_host_env(layout)}, input=None)
    return code


def _fetch(executor: RemoteExecutor, run_id: str, results: Path, log: Callable[[str], None]) -> None:
    """Copy the run directory back, failed runs included; a lost session is only logged."""
    run_dir = f"{executor.layout.runs}/{run_id}"
    exists = [executor.layout.python, "-c", "import os, sys; sys.exit(not os.path.isdir(sys.argv[1]))", run_dir]
    try:
        code, _ = executor.run("fetch", exists, env={}, input=None)
    except HostUnavailable as error:
        log(f"cannot fetch {run_dir}: {error}")
        return
    if code == 0:
        sync.fetch(executor.remote, run_dir, results)
        log(f"fetched {executor.remote.alias}:{run_dir} -> {results}")


def _mirror(
    executor: RemoteExecutor,
    name: str,
    local: Path,
    paths: list[str],
    destination: str,
    log: Callable[[str], None],
) -> None:
    """Push `paths` of `local` to `destination` and prune what they no longer list."""
    remote = executor.remote
    started = time.monotonic()
    make = "import os, sys; os.makedirs(sys.argv[1], exist_ok=True)"
    _step(executor, "sync", [remote.python, "-c", make, destination])  # rsync makes no parents
    sync.push(remote, paths, local, destination)
    _step(
        executor,
        "sync",
        [remote.python, remote.agent, "prune", "--dir", destination],
        input="\n".join(paths) + "\n",
    )
    log(f"sync {name}: {len(paths)} files in {time.monotonic() - started:.1f} s")


def _print_sink(step: str, stream: str, text: str) -> None:
    sys.stderr.write(f"[{step}] {text}" if text.endswith("\n") else f"[{step}] {text}\n")


def _print_log(message: str) -> None:
    print(f"magnitude-remote: {message}", file=sys.stderr, flush=True)


def execute(
    job: Job,
    host: str | None = None,
    *,
    hosts: Hosts | None = None,
    sink: OutputSink = _print_sink,
    log: Callable[[str], None] = _print_log,
) -> Outcome:
    """Run `job` on `host` (an ssh alias or pool) or, with no host, locally."""
    hosts = hosts or Hosts.load()
    build = identify()  # of the tree being synchronized; hosts never need git
    run_id = new_run_id(job.tool)
    results = RESULTS / run_id
    if host is None:
        executor = LocalExecutor(hosts.defaults, sink)
        return Outcome(run_id, None, results, perform(executor, job, run_id, build, log))
    with acquire(hosts.resolve(host), run=run_id, max_hold=job.max_hold_seconds, log=log) as session:
        executor = RemoteExecutor(session, sink)
        try:
            _mirror(executor, "src", sync.INFERENCE, sync.manifest(), executor.layout.src, log)
            for tree in job.trees:
                _mirror(executor, tree.name, tree.local, list(tree.paths), executor.tree(tree), log)
            code = perform(executor, job, run_id, build, log)
        finally:
            _fetch(executor, run_id, results, log)
        return Outcome(run_id, executor.alias, results, code)
