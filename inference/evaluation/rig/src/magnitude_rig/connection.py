"""Connecting to a host: OS probe, agent deployment, and the agent session holding a run's holds."""

import base64
import json
import queue
import shlex
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .hosts import HostConfig

AGENT_SOURCE = (Path(__file__).parent / "host" / "agent.py").read_bytes()
HEARTBEAT_SECONDS = 10.0
# An unreachable host must not hold up placement among the reachable ones: a reachable host
# connects in well under a second, so a few seconds bounds the cost of one that does not.
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
_BOOTSTRAP = """
import base64, hashlib, json, os, sys
root = os.path.abspath(os.path.expanduser(sys.argv[1]))
models = os.path.abspath(os.path.expanduser(sys.argv[2]))
env = {{name: os.path.expanduser(value) for name, value in json.loads(sys.argv[3]).items()}}
source = base64.b64decode({source!r})
agent = os.path.join(root, "agent", hashlib.sha256(source).hexdigest()[:16] + ".py")
for directory in (os.path.dirname(agent), os.path.join(root, "src"), models):  # src: steps' cwd
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError:
        sys.exit("cannot create %s; run `mrig setup <host>` once" % directory)
if not os.path.exists(agent):
    temporary = "%s.%d.tmp" % (agent, os.getpid())
    with open(temporary, "wb") as stream:
        stream.write(source)
    os.replace(temporary, agent)
import glob, platform, shutil
backends = ["cpu"]
if platform.system() == "Darwin" and platform.machine() == "arm64":
    backends.append("metal")
if shutil.which("nvidia-smi") or os.path.exists("/dev/nvidiactl"):
    backends.append("cuda")
icds = [p for p in env.get("VK_ICD_FILENAMES", "").split(os.pathsep) if p]
icds += glob.glob("/usr/share/vulkan/icd.d/*.json") + glob.glob("/etc/vulkan/icd.d/*.json")
if any(os.path.exists(p) for p in icds):
    backends.append("vulkan")
ready = os.path.exists(os.path.join(root, "rig", "ready.json"))
print(json.dumps({{"root": root, "models": models, "agent": agent, "env": env,
                  "backends": backends, "ready": ready}}))
"""


def shell(alias: str, local: bool, line: str) -> list[str]:
    """The argv running a shell command line on a host: over ssh, or here for this machine."""
    return ["/bin/sh", "-c", line] if local else ["ssh", *SSH_OPTIONS, alias, line]


class HostUnavailable(Exception):
    """The host cannot be reached, or its agent session ended."""


@dataclass(frozen=True)
class Remote:
    """A connected host: its probed platform and resolved absolute paths."""

    alias: str
    local: bool
    windows: bool
    system: str
    machine: str
    python: str
    root: str
    models: str
    agent: str
    env: dict[str, str]  # the host's own step environment, `~` expanded on the host
    backends: tuple[str, ...]  # what the host can run, probed on connect
    ready: bool  # `mrig setup` verified the host

    def command(self, argv: list[str], *, login: bool = False) -> list[str]:
        """The ssh argv running `argv` on the host (in a login shell when asked, on POSIX)."""
        if self.windows:
            return ["ssh", *SSH_OPTIONS, self.alias, subprocess.list2cmdline(argv)]
        line = shlex.join(argv)
        if login:
            line = f'exec "$SHELL" -lc {shlex.quote("exec " + line)}'
        return shell(self.alias, self.local, line)

    def place(self, path: str) -> str:
        """`path` on the host as rsync names it."""
        return path if self.local else f"{self.alias}:{path}"

    def path(self, *parts: str) -> str:
        return "/".join((self.root, *parts))


def _attempt(host, line: str, stdin: bytes | None, timeout: float):
    """Run `line` on `host` (a HostConfig or Remote); an unreachable host (ssh's own 255 or a
    timeout) raises."""
    alias = host.alias
    try:
        result = subprocess.run(
            shell(alias, host.local, line), input=stdin, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise HostUnavailable(f"{alias}: ssh timed out after {timeout} s") from error
    if result.returncode == 255 and not host.local:
        raise HostUnavailable(f"{alias}: {result.stderr.decode(errors='replace').strip()}")
    return result


def read(host, line: str, *, timeout: float = 120) -> tuple[int, str, str]:
    """Run a read-only command line on the host: its exit code, stdout and stderr."""
    result = _attempt(host, line, None, timeout)
    return result.returncode, result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")


def _ssh(host, line: str, *, stdin: bytes | None = None, timeout: float = 60) -> str:
    result = _attempt(host, line, stdin, timeout)
    if result.returncode != 0:
        raise HostUnavailable(
            f"{host.alias}: `{line[:80]}` exited {result.returncode}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode()


def connect(config: HostConfig) -> Remote:
    """Probe the host's OS (`uname`; failure means Windows) and deploy the agent."""
    probe = _attempt(config, "uname -sm", None, 30)
    if probe.returncode == 0:
        system, machine = probe.stdout.decode().split()
        windows, python = False, "python3"
    else:
        system, machine, windows, python = "Windows", "unknown", True, "python"
    bootstrap = _BOOTSTRAP.format(source=base64.b64encode(AGENT_SOURCE).decode())
    argv = [python, "-", config.root, config.models, json.dumps(dict(config.env))]
    line = subprocess.list2cmdline(argv) if windows else shlex.join(argv)
    probed = json.loads(_ssh(config, line, stdin=bootstrap.encode()).strip().splitlines()[-1])
    probed["backends"] = tuple(probed["backends"])
    return Remote(config.alias, config.local, windows, system, machine, python, **probed)


SETUP_TOOLS = ("cargo", "rustc", "uv", "rsync")


def setup(config: HostConfig) -> str:
    """Prepare and verify a host (POSIX): create the cache root and model store owned by the
    ssh user, through passwordless sudo where their parent is not writable (for example a
    root-owned scratch volume, recreated whenever ephemeral instance storage is), then check
    every tool runs need on the job PATH and record the host as ready only when all are there.
    Placement uses only ready hosts."""

    def shell_path(path: str) -> str:
        return '"$HOME"/' + shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)

    root = shell_path(config.root)
    directories = " ".join(shell_path(p) for p in (config.root, config.models))
    tools = " ".join(SETUP_TOOLS)
    script = (
        'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; set -e; '
        f"for d in {directories}; do "
        'if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then echo "$d: ready"; '
        'else sudo -n install -d -o "$(id -un)" -g "$(id -gn)" "$d"; echo "$d: created with sudo"; fi; '
        "done; missing=; "
        f"for t in {tools}; do "
        'v=$("$t" --version 2>/dev/null | head -1); '
        'if [ -n "$v" ]; then echo "$t: $v"; '
        'else echo "$t: MISSING"; missing="$missing $t"; fi; done; '
        f"mkdir -p {root}/rig; "
        f'if [ -z "$missing" ]; then date -u +%FT%TZ > {root}/rig/ready.json; echo "ready"; '
        f'else rm -f {root}/rig/ready.json; echo "not ready: missing$missing"; fi'
    )
    return _ssh(config, f"sh -c {shlex.quote(script)}")


def host_status(remote: Remote, checkout: str | None = None) -> dict:
    """The host agent's `status`: live holds and requests, queue estimates, free disk, and
    whether `checkout` is present."""
    argv = [remote.python, remote.agent, "status", "--root", remote.root]
    if checkout:
        argv += ["--checkout", checkout]
    return json.loads(_ssh(remote, remote.command(argv)[-1]).strip().splitlines()[-1])


class AgentSession:
    """One long-lived ssh session to the host agent for one run, kept alive by a heartbeat."""

    def __init__(self, remote: Remote, *, who: str, run: str, purpose: str, estimate: float):
        self.remote = remote
        argv = [
            remote.python, remote.agent, "session", "--root", remote.root, "--who", who,
            "--run", run, "--purpose", purpose, "--estimate", str(estimate),
        ]
        self._process = subprocess.Popen(
            remote.command(argv, login=True),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._events: queue.Queue = queue.Queue()
        self._write_lock = threading.Lock()
        self._closed = threading.Event()
        self._stderr: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _read(self) -> None:
        for line in self._process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue  # login-shell noise
            if isinstance(event, dict) and "event" in event:
                self._events.put(event)
        self._events.put(None)

    def _read_stderr(self) -> None:
        for line in self._process.stderr:
            self._stderr.append(line)

    def _send(self, request: dict) -> None:
        with self._write_lock:
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def _heartbeat(self) -> None:
        while not self._closed.wait(HEARTBEAT_SECONDS):
            self._send({"op": "heartbeat"})

    def _ended(self) -> HostUnavailable:
        self._process.wait()
        detail = "".join(self._stderr).strip()
        return HostUnavailable(f"{self.remote.alias}: agent session ended. {detail}".strip())

    def _next(self) -> dict:
        event = self._events.get()
        if event is None:
            self._events.put(None)
            raise self._ended()
        if event["event"] == "stopped":
            raise HostUnavailable(f"{self.remote.alias}: agent stopped: {event['reason']}")
        if event["event"] == "error":
            raise RuntimeError(f"{self.remote.alias} agent: {event['message']}")
        return event

    def hold(self, kind: str, on_wait: Callable[[list[dict]], None]) -> None:
        """Return once the agent holds `kind`, reporting each change of what blocks it."""
        self._send({"op": "hold", "kind": kind})
        while True:
            event = self._next()
            if event["event"] == "held" and event["kind"] == kind:
                return
            if event["event"] == "waiting":
                on_wait(event["blocking"])

    def release(self, kind: str) -> None:
        self._send({"op": "release", "kind": kind})
        while self._next()["event"] != "released":
            pass

    def kill(self) -> None:
        """End the running command, if any (its output and exit are drained)."""
        self._send({"op": "kill"})
        while self._next()["event"] != "killed":
            pass

    def evict(self, keep: str) -> dict:
        self._send({"op": "evict", "keep": keep})
        while (event := self._next())["event"] != "evicted":
            pass
        return event

    def run(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        input: str | None,
        on_output: Callable[[str, str], None],
    ) -> int:
        self._send({"op": "run", "argv": argv, "cwd": cwd, "env": env, "input": input})
        while True:
            event = self._next()
            if event["event"] == "output":
                on_output(event["stream"], event["text"])
            elif event["event"] == "exited":
                return event["code"]

    def close(self) -> None:
        self._send({"op": "end"})
        self._closed.set()
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
