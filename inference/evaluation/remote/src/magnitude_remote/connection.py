"""Connecting to a host: OS probe, agent deployment, and the lock-holding agent session."""

import base64
import json
import queue
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .hosts import HostConfig

AGENT_SOURCE = (Path(__file__).parent / "host" / "agent.py").read_bytes()
HEARTBEAT_SECONDS = 10.0
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
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
        sys.exit("cannot create %s; run `magnitude-remote setup <host>` once" % directory)
if not os.path.exists(agent):
    temporary = "%s.%d.tmp" % (agent, os.getpid())
    with open(temporary, "wb") as stream:
        stream.write(source)
    os.replace(temporary, agent)
print(json.dumps({{"root": root, "models": models, "agent": agent, "env": env}}))
"""


class HostUnavailable(Exception):
    """The host cannot be reached, or its agent session ended."""


@dataclass(frozen=True)
class Remote:
    """A connected host: its probed platform and resolved absolute paths."""

    alias: str
    windows: bool
    system: str
    machine: str
    python: str
    root: str
    models: str
    agent: str
    env: dict[str, str]  # the host's own step environment, `~` expanded on the host

    def command(self, argv: list[str], *, login: bool = False) -> list[str]:
        """The ssh argv running `argv` on the host (in a login shell when asked, on POSIX)."""
        if self.windows:
            return ["ssh", *SSH_OPTIONS, self.alias, subprocess.list2cmdline(argv)]
        line = shlex.join(argv)
        if login:
            line = f'exec "$SHELL" -lc {shlex.quote("exec " + line)}'
        return ["ssh", *SSH_OPTIONS, self.alias, line]

    def path(self, *parts: str) -> str:
        return "/".join((self.root, *parts))


def _attempt(alias: str, line: str, stdin: bytes | None, timeout: float):
    """Run `line` over ssh; an unreachable host (ssh's own 255 or a timeout) raises."""
    try:
        result = subprocess.run(
            ["ssh", *SSH_OPTIONS, alias, line], input=stdin, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise HostUnavailable(f"{alias}: ssh timed out after {timeout} s") from error
    if result.returncode == 255:
        raise HostUnavailable(f"{alias}: {result.stderr.decode(errors='replace').strip()}")
    return result


def _ssh(alias: str, line: str, *, stdin: bytes | None = None, timeout: float = 60) -> str:
    result = _attempt(alias, line, stdin, timeout)
    if result.returncode != 0:
        raise HostUnavailable(
            f"{alias}: `{line[:80]}` exited {result.returncode}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode()


def connect(config: HostConfig) -> Remote:
    """Probe the host's OS (`uname`; failure means Windows) and deploy the agent."""
    probe = _attempt(config.alias, "uname -sm", None, 30)
    if probe.returncode == 0:
        system, machine = probe.stdout.decode().split()
        windows, python = False, "python3"
    else:
        system, machine, windows, python = "Windows", "unknown", True, "python"
    bootstrap = _BOOTSTRAP.format(source=base64.b64encode(AGENT_SOURCE).decode())
    argv = [python, "-", config.root, config.models, json.dumps(dict(config.env))]
    line = subprocess.list2cmdline(argv) if windows else shlex.join(argv)
    paths = json.loads(_ssh(config.alias, line, stdin=bootstrap.encode()).strip().splitlines()[-1])
    return Remote(config.alias, windows, system, machine, python, **paths)


def setup(config: HostConfig) -> str:
    """One-time host preparation (POSIX): create the cache root and model store owned by the
    ssh user, through passwordless sudo where their parent is not writable (for example a
    a root-owned scratch volume, recreated whenever ephemeral instance storage is)."""

    def shell_path(path: str) -> str:
        return '"$HOME"/' + shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)

    directories = " ".join(shell_path(p) for p in (config.root, config.models))
    script = (
        f"set -e; for d in {directories}; do "
        'if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then echo "$d: ready"; '
        'else sudo -n install -d -o "$(id -un)" -g "$(id -gn)" "$d"; echo "$d: created with sudo"; fi; '
        "done"
    )
    return _ssh(config.alias, script)


def holder_status(remote: Remote) -> dict:
    line = remote.command([remote.python, remote.agent, "status", "--root", remote.root])[-1]
    return json.loads(_ssh(remote.alias, line).strip().splitlines()[-1])


class AgentSession:
    """One long-lived ssh session to the host agent, kept alive by a heartbeat thread."""

    def __init__(self, remote: Remote, *, who: str, run: str, max_hold: float):
        self.remote = remote
        argv = [
            remote.python, remote.agent, "hold", "--root", remote.root,
            "--who", who, "--run", run, "--max-hold", str(max_hold),
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
        self.holder: dict | None = None

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

    def _next(self, timeout: float | None) -> dict:
        event = self._events.get(timeout=timeout)
        if event is None:
            self._events.put(None)
            raise self._ended()
        if event["event"] == "stopped":
            raise HostUnavailable(f"{self.remote.alias}: agent stopped: {event['reason']}")
        return event

    def poll_locked(self, timeout: float, on_wait: Callable[[dict | None, int], None]) -> bool:
        """True once the agent holds the lock; reports changes of the holder and of how many
        queued sessions are ahead of this one."""
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                event = self._next(remaining)
            except queue.Empty:
                return False
            if event["event"] == "locked":
                self.holder = event["holder"]
                return True
            if event["event"] == "waiting":
                on_wait(event["holder"], event["ahead"])
        return False

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
            event = self._next(None)
            if event["event"] == "output":
                on_output(event["stream"], event["text"])
            elif event["event"] == "exited":
                return event["code"]
            elif event["event"] == "error":
                raise RuntimeError(f"{self.remote.alias} agent: {event['message']}")

    def close(self) -> None:
        self._send({"op": "release"})
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
