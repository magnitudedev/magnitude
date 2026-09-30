"""Owned child lifetime, progress supervision and a small adapter boundary around real
serving processes."""

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

from ..fixtures.contexts import Context, Counter
from ..session_bench.models import Artifact, Target
from ..session_bench.options import EngineOptions, Watchdog
from ..session_bench.policy import CONTEXT_ALIGNMENT, STARTUP_TIMEOUT_SECONDS
from ..session_bench.results import RunStore
from ..session_bench.sessions import Plan, Request, encoded


async def command(args: list[str], cwd: Path, log: Path, *, env: dict | None = None) -> str:
    """Subprocess output goes to evidence; cancellation always retires the process group."""
    with log.open("ab") as output:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            async with asyncio.timeout(STARTUP_TIMEOUT_SECONDS):
                code = await process.wait()
            if code:
                raise RuntimeError(f"command failed ({code}); see {log}")
        finally:
            await stop(process)
    return log.read_text(errors="replace")


async def stop(process: asyncio.subprocess.Process):
    # The group can outlive its leader. Always retire descendants as well.
    for sig, deadline in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), deadline)
        except TimeoutError:
            continue
        # The supervisor exits only after reaping the server it owns. Once it
        # has exited successfully, the process group is retired; signalling
        # that numeric group again can race PID/group reuse on Darwin.
        break
    await asyncio.wait_for(process.wait(), 5)


def memory_bytes(pid: int) -> int:
    total = 0
    try:
        parent = psutil.Process(pid)
        for process in [parent, *parent.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total


def runtime_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        p for p in (root / "src").rglob("*") if p.suffix in (".py", ".json") and p.is_file()
    )
    files += [p for p in (root / "pyproject.toml", root / "uv.lock") if p.is_file()]
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def installed_versions(root: Path) -> dict[str, str]:
    sites = list((root / ".venv" / "lib").glob("python*/site-packages"))
    if not sites:
        raise ValueError(f"frozen Python environment missing: {root}")
    return {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions(path=[str(p) for p in sites])
    }


@dataclass
class Running:
    endpoint: str
    model: str
    process: asyncio.subprocess.Process
    readiness: dict
    baseline_bytes: int
    peak_bytes: int
    watchdog: Watchdog
    log: Path
    # Supervision arms at the first request; preparation launches are never bounded.
    active: dict[str, float] = field(default_factory=dict)
    progressed_at: float | None = None
    logged_bytes: int = 0
    retired: str | None = None

    def progress(self) -> None:
        self.progressed_at = time.monotonic()

    def started(self, request: str) -> None:
        self.active[request] = time.monotonic()
        self.progress()

    def finished(self, request: str) -> None:
        # Requests that never started (dependency failures, interruption) also settle here.
        self.active.pop(request, None)
        self.progress()

    def violation(self) -> str | None:
        """Observe engine output as progress, then report an exceeded watchdog bound."""
        if not self.watchdog.enabled or self.progressed_at is None:
            return None
        now = time.monotonic()
        size = self.log.stat().st_size
        if size != self.logged_bytes:
            self.logged_bytes = size
            self.progressed_at = now
        stall = self.watchdog.stall_seconds
        if stall is not None and now - self.progressed_at >= stall:
            active = ",".join(sorted(self.active)) or "none"
            return f"no progress for {now - self.progressed_at:.1f}s (active: {active})"
        limit = self.watchdog.request_seconds
        if limit is not None and self.active:
            request, since = min(self.active.items(), key=lambda item: item[1])
            if now - since >= limit:
                return f"request {request} exceeded {limit:g}s"
        return None


class Adapter:
    timing_basis = "native-model-service"
    extensions: dict = {}

    def __init__(
        self,
        root: Path,
        target: Target,
        artifact: Artifact,
        store: RunStore,
        options: EngineOptions,
    ):
        self.root, self.target, self.artifact, self.store = root, target, artifact, store
        self.options = options
        self.runtime: Path | None = None
        self.identity = {}
        self.source_identity = ""

    def environment(self) -> dict:
        env = os.environ.copy()
        env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=str(self.root / "src"),
        )
        return env

    async def prepare(self) -> None:
        if self.runtime:
            await command(
                ["uv", "sync", "--locked"],
                self.runtime,
                self.store.path / "logs" / f"{self.target.id}-install.log",
            )
            self.identity = self.store.snapshot(self.target.id, self.runtime)
            self.source_identity = runtime_digest(self.runtime)
            version_path = self.store.path / f"{self.target.id}-versions.json"
            code = (
                "import importlib.metadata,json,pathlib; "
                "pathlib.Path(" + repr(str(version_path)) + ").write_text(json.dumps({"
                "d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))"
            )
            await command(
                ["uv", "run", "--frozen", "--no-sync", "python", "-c", code],
                self.runtime,
                self.store.path / "logs" / f"{self.target.id}-versions.log",
            )
            self.identity["dependencies"] = json.loads(version_path.read_text())

    def verify(self) -> None:
        self.artifact.verify_unchanged()
        if self.runtime:
            if (
                runtime_digest(self.runtime) != self.source_identity
                or installed_versions(self.runtime) != self.identity["dependencies"]
            ):
                raise ValueError(f"engine source or environment changed: {self.target.engine}")

    def argv(self, port: int, context: int, parallel: int, directory: Path) -> list[str]:
        raise NotImplementedError

    def served_model(self) -> str:
        return "session-bench"

    def ready_path(self) -> str:
        return "/health"

    def verify_ready(self, data: dict, context: int, parallel: int) -> None:
        raise NotImplementedError

    @contextlib.asynccontextmanager
    async def launch(self, context: int, parallel: int, label: str) -> AsyncIterator[Running]:
        self.verify()
        directory = self.store.path / "engines" / f"{self.target.id}-{label}"
        directory.mkdir(parents=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = self.argv(port, context, parallel, directory)
        self.store.event(
            "launch",
            target=self.target.id,
            label=label,
            argv=args,
            context_capacity=context,
            parallel_sequences=parallel,
        )
        log = self.store.path / "logs" / f"{self.target.id}-{label}.log"
        with log.open("wb") as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).with_name("supervise.py")),
                str(os.getpid()),
                *args,
                cwd=self.runtime or self.root,
                env=self.environment(),
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            running = Running(
                f"http://127.0.0.1:{port}",
                self.served_model(),
                process,
                {},
                0,
                0,
                self.options.watchdog,
                log,
            )

            async def sample():
                sample_index = 0
                while True:
                    if reason := running.violation():
                        running.retired = reason
                        self.store.event(
                            "watchdog", target=self.target.id, label=label, reason=reason
                        )
                        # The runner observes the exit and fails the run with this reason.
                        await stop(process)
                        return
                    memory = memory_bytes(process.pid)
                    running.peak_bytes = max(running.peak_bytes, memory)
                    self.store.record_stream(
                        "memory.jsonl",
                        {
                            "target": self.target.id,
                            "label": label,
                            "at": time.time(),
                            "rss_bytes": memory,
                        },
                    )
                    if running.readiness and sample_index % 4 == 0:
                        try:
                            allocation = await self.memory_observation(running)
                        except (httpx.HTTPError, ValueError) as error:
                            self.store.event(
                                "allocation_sample_failed",
                                target=self.target.id,
                                label=label,
                                error=str(error),
                            )
                        else:
                            if allocation is not None:
                                self.store.append(
                                    "allocations.jsonl",
                                    {
                                        "target": self.target.id,
                                        "label": label,
                                        "at": time.time(),
                                        "observation": allocation,
                                    },
                                )
                    sample_index += 1
                    await asyncio.sleep(0.25)

            sampling = asyncio.create_task(sample())
            try:
                async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                    async with asyncio.timeout(STARTUP_TIMEOUT_SECONDS):
                        while True:
                            if process.returncode is not None:
                                raise RuntimeError(f"engine exited before readiness; see {log}")
                            try:
                                response = await client.get(running.endpoint + self.ready_path())
                                if response.status_code == 200:
                                    data = response.json()
                                    if self.verify_ready(data, context, parallel) is False:
                                        await asyncio.sleep(0.25)
                                        continue
                                    # Verify listener ownership even if the ephemeral port raced.
                                    owner = psutil.Process(process.pid)
                                    members = [owner, *owner.children(recursive=True)]
                                    if not any(
                                        c.status == psutil.CONN_LISTEN and c.laddr.port == port
                                        for p in members
                                        for c in p.net_connections("tcp")
                                    ):
                                        raise RuntimeError(
                                            "readiness listener is not owned by engine"
                                        )
                                    running.readiness = data
                                    break
                                if response.status_code != 503:
                                    raise RuntimeError(f"readiness failed: {response.text[:2000]}")
                            except (httpx.ConnectError, httpx.TimeoutException):
                                pass
                            await asyncio.sleep(0.25)
                running.baseline_bytes = memory_bytes(process.pid)
                self.store.event(
                    "ready",
                    target=self.target.id,
                    label=label,
                    evidence=running.readiness,
                    timing_basis=self.timing_basis,
                )
                yield running
            finally:
                sampling.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sampling
                await stop(process)
                self.store.event(
                    "stopped", target=self.target.id, label=label, peak_rss_bytes=running.peak_bytes
                )

    async def prompt_counts(self, plan: Plan) -> dict[str, int]:
        raise NotImplementedError

    async def memory_observation(self, engine: Running) -> dict | None:
        """Optional allocation census after the measured requests finish."""
        return None

    def context_plan(self, context: Context) -> Plan:
        return Plan(
            requests=(
                Request(
                    id="fixture-sizing",
                    section="context",
                    session="fixture-sizing",
                    checkpoint=0,
                    fixture_id="tools.bfcl" if context.tools else "prose.moby-dick",
                    workload="tools" if context.tools else "prose",
                    messages=context.messages,
                    tools=context.tools,
                    expected=[],
                ),
            ),
            parallel_sequences=1,
            corpus_digest="preparation",
        )

    @contextlib.asynccontextmanager
    async def context_counter(self) -> AsyncIterator[Counter]:
        async def count(context: Context) -> int:
            return (await self.prompt_counts(self.context_plan(context)))["fixture-sizing"]

        yield count

    def provisional_capacity(self, plan: Plan) -> int:
        # Only a launch allowance for native tokenization, not a prompt-token measurement.
        size = max(
            len(encoded(r.body(self.served_model())).encode()) for r in plan.prepared_requests
        )
        needed = size * 2 + max(r.output_limit for r in plan.prepared_requests)
        aligned = (needed + CONTEXT_ALIGNMENT - 1) // CONTEXT_ALIGNMENT * CONTEXT_ALIGNMENT
        return min(self.artifact.context_limit // CONTEXT_ALIGNMENT * CONTEXT_ALIGNMENT, aligned)
