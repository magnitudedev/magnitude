"""Connecting to a host: OS probe, host package deployment, and a run's runner on the host."""

import base64
import contextlib
import io
import json
import shlex
import subprocess
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .host.agent import MODULE as AGENT
from .hosts import HostConfig


def _host_package() -> bytes:
    """The host package (`magnitude_rig.host`, the agent among it) as a zip any Python 3.9+
    runs as `python ZIP MODULE ARGS...`. Its bytes depend only on the sources, so their hash
    names it on the host."""
    package = Path(__file__).parent
    files = {"__main__.py": b"import runpy, sys\n"
             b"runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)\n"}
    for path in (package / "__init__.py", *sorted((package / "host").glob("*.py"))):
        files[path.relative_to(package.parent).as_posix()] = path.read_bytes()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        for name, data in files.items():
            stream.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
    return archive.getvalue()


HOST_PACKAGE = _host_package()
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
package = os.path.join(root, "agent", hashlib.sha256(source).hexdigest()[:16] + ".zip")
for directory in (os.path.dirname(package), os.path.join(root, "src"), models):  # src: steps' cwd
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError:
        sys.exit("cannot create %s; run `mrig setup <host>` once" % directory)
if not os.path.exists(package):
    temporary = "%s.%d.tmp" % (package, os.getpid())
    with open(temporary, "wb") as stream:
        stream.write(source)
    os.replace(temporary, package)
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
print(json.dumps({{"root": root, "models": models, "package": package, "env": env,
                  "backends": backends, "ready": ready}}))
"""


def shell(alias: str, local: bool, line: str) -> list[str]:
    """The argv running a shell command line on a host: over ssh, or here for this machine."""
    return ["/bin/sh", "-c", line] if local else ["ssh", *SSH_OPTIONS, alias, line]


def windows_command(argv: list[str], cwd: str | None = None) -> str:
    """Preserve native arguments across either Windows OpenSSH default shell."""
    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"
    script = "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; "
    script += "$start = New-Object System.Diagnostics.ProcessStartInfo; $start.UseShellExecute = $false; "
    script += f"$start.FileName = {quote(argv[0])}; $start.Arguments = {quote(subprocess.list2cmdline(argv[1:]))}; "
    if cwd is not None:
        script += f"$start.WorkingDirectory = {quote(cwd)}; "
    script += "$child = [System.Diagnostics.Process]::Start($start); $child.WaitForExit(); exit $child.ExitCode"
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return f"powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded}"


class HostError(Exception):
    """A host did not do what was asked: it could not be reached, or a command on it failed."""


class HostUnavailable(HostError):
    """The host cannot be reached: ssh's own exit 255 or a timeout, which a later attempt may
    pass."""


class CommandFailed(HostError):
    """A command on a reachable host failed, which no later attempt changes."""


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
    package: str  # the deployed host package (`host`)
    env: dict[str, str]  # the host's own step environment, `~` expanded on the host
    backends: tuple[str, ...]  # what the host can run, probed on connect
    ready: bool  # `mrig setup` verified the host
    temperatures: bool = True  # the host exposes temperature sensors (`HostConfig.temperatures`)

    def command(self, argv: list[str], *, login: bool = False, cwd: str | None = None) -> list[str]:
        """The ssh argv running `argv` on the host (in a login shell when asked, on POSIX)."""
        if self.windows:
            return ["ssh", *SSH_OPTIONS, self.alias, windows_command(argv, cwd)]
        line = shlex.join(argv)
        if cwd is not None:
            line = f"cd {shlex.quote(cwd)} && {line}"
        if login:
            line = f'exec "$SHELL" -lc {shlex.quote("exec " + line)}'
        return shell(self.alias, self.local, line)

    def host(self, module: str, *arguments: str) -> list[str]:
        """The argv running a module of the deployed host package, e.g. `magnitude_rig.host.agent`."""
        return [self.python, self.package, module, *arguments]

    def place(self, path: str) -> str:
        """`path` on the host as rsync names it."""
        return path if self.local else f"{self.alias}:{path}"

    def path(self, *parts: str) -> str:
        return "/".join((self.root, *parts))


def unreachable(host, returncode: int) -> bool:
    """Whether a command run on `host` (a HostConfig or Remote) through ssh exited with ssh's own
    255, which means the host was not reached, rather than the command's code."""
    return returncode == 255 and not host.local


def _attempt(host, line: str, stdin: bytes | None, timeout: float):
    """Run `line` on `host` (a HostConfig or Remote); an unreachable host (ssh's own 255 or a
    timeout) raises."""
    alias = host.alias
    try:
        result = subprocess.run(
            shell(alias, host.local, line), input=stdin, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise HostUnavailable(f"{alias}: unreachable (ssh timed out after {timeout} s)") from error
    if unreachable(host, result.returncode):
        raise HostUnavailable(f"{alias}: unreachable ({result.stderr.decode(errors='replace').strip()})")
    return result


def reachable(host) -> bool:
    """Whether `host` (a HostConfig or Remote) answers now."""
    try:
        _attempt(host, "true", None, 30)
    except HostUnavailable:
        return False
    return True


def read(host, line: str, *, timeout: float = 120) -> tuple[int, str, str]:
    """Run a read-only command line on the host: its exit code, stdout and stderr."""
    result = _attempt(host, line, None, timeout)
    return result.returncode, result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")


def _ssh(host, line: str, *, stdin: bytes | None = None, timeout: float = 60) -> str:
    result = _attempt(host, line, stdin, timeout)
    if result.returncode != 0:
        raise CommandFailed(
            f"{host.alias}: `{line[:80]}` exited {result.returncode}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode()


def connect(config: HostConfig) -> Remote:
    """Probe the host's OS (`uname`; failure means Windows) and deploy the host package."""
    probe = _attempt(config, "uname -sm", None, 30)
    if probe.returncode == 0:
        system, machine = probe.stdout.decode().split()
        windows, python = False, "python3"
    else:
        system, machine, windows, python = "Windows", "unknown", True, "python"
    bootstrap = _BOOTSTRAP.format(source=base64.b64encode(HOST_PACKAGE).decode())
    argv = [python, "-", config.root, config.models, json.dumps(dict(config.env))]
    line = windows_command(argv) if windows else shlex.join(argv)
    probed = json.loads(_ssh(config, line, stdin=bootstrap.encode()).strip().splitlines()[-1])
    probed["backends"] = tuple(probed["backends"])
    return Remote(
        config.alias, config.local, windows, system, machine, python, **probed,
        temperatures=config.temperatures,
    )


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


def host_status(remote: Remote, sides: list[tuple[str, str, str]] = ()) -> dict:
    """The host agent's `status`: live holds and requests, queue estimates, free disk, and, for
    a run whose sides are `sides` (each its source's key, its tree's, `engine.tree_key`, and its
    build's, `engine.build_key`), the checkout each would claim or share now, what its build
    would need and the room it would find."""
    argv = remote.host(AGENT, "status", "--root", remote.root)
    for source, tree, build in sides:
        argv += ["--source", source, "--tree", tree, "--build", build]
    return json.loads(_ssh(remote, remote.command(argv)[-1]).strip().splitlines()[-1])


def run_state(remote: Remote, run: str) -> str:
    """How run `run` stands on the host (the agent's `state`): `running`, `ended`, `vanished` or
    `absent`. Read through this command's agent, whichever agent the runner runs under."""
    argv = remote.host(AGENT, "state", "--root", remote.root, "--run", run)
    return _json_line(_ssh(remote, remote.command(argv)[-1]))["state"]


class HostRun:
    """One run's runner on a host, through the agent it runs under (`host/agent.py`, its
    `invocation`): each call is its own ssh command, so none outlives a lost connection."""

    def __init__(self, remote: Remote, run: str, agent: list[str]):
        self.remote, self.run, self.agent = remote, run, agent

    @classmethod
    @contextlib.contextmanager
    def staging(cls, remote: Remote, run: str) -> Iterator["Callable[[dict], HostRun] | HostRun"]:
        """The host's hold on the run's directory while the caller stages it (the agent's
        `start`), for the block: it yields `start(plan)`, which starts the run's runner with
        `plan` (it owns the run once that returns). Leaving the block without starting
        withdraws, as does a lost connection: the directory is then a gone run's. For a run an
        earlier attempt started (its reply lost), it yields that run's runner instead: nothing
        is staged over it."""
        argv = remote.host(AGENT, "start", "--root", remote.root, "--run", run)
        process = subprocess.Popen(remote.command(argv, login=True), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def failed(what: str) -> HostError:
            process.wait()
            failure = HostUnavailable if unreachable(remote, process.returncode) else CommandFailed
            return failure(f"{remote.alias}: {what} run {run} exited {process.returncode}: "
                           f"{process.stderr.read().strip()}")

        def start(plan: dict) -> "HostRun":
            stdout, stderr = process.communicate(json.dumps(plan))
            if process.returncode != 0:
                failure = HostUnavailable if unreachable(remote, process.returncode) else CommandFailed
                raise failure(f"{remote.alias}: starting run {run} exited {process.returncode}: {stderr.strip()}")
            return cls(remote, run, _json_line(stdout)["agent"])

        try:
            for line in process.stdout:  # past login-shell noise to the hold
                try:
                    held = json.loads(line)
                except ValueError:
                    continue
                if isinstance(held, dict) and ("staging" in held or "started" in held):
                    break
            else:
                raise failed("holding")
            yield cls(remote, run, held["started"]["agent"]) if "started" in held else start
        finally:
            if process.poll() is None:
                process.stdin.close()  # withdraw: no plan
                process.wait()

    def _argv(self, command: str, *extra: str) -> list[str]:
        return [self.remote.python, *self.agent, command, "--root", self.remote.root, "--run", self.run, *extra]

    def follow(self, after: int) -> Iterator[dict]:
        """The run's events after sequence `after`, as the runner writes them, through `ended`
        (or `vanished`); a lost connection raises `HostUnavailable`, the agent's own failure
        `CommandFailed`."""
        process = subprocess.Popen(
            self.remote.command(self._argv("follow", "--after", str(after))),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # login-shell noise
                if not isinstance(event, dict) or "event" not in event:
                    continue  # login-shell noise that parses as JSON (a bare number)
                yield event
                if event["event"] in ("ended", "vanished"):
                    return
            process.wait()
            failure = HostUnavailable if unreachable(self.remote, process.returncode) else CommandFailed
            raise failure(f"{self.remote.alias}: following run {self.run} ended: "
                          f"{process.stderr.read().strip()}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()

    def stop(self) -> dict:
        """Stop the runner if it still runs (the agent's `stop`): whether it was stopped, and
        why not; the agent's own failure raises `CommandFailed`."""
        return _json_line(_ssh(self.remote, self.remote.command(self._argv("stop"))[-1]))

    def fetched(self) -> None:
        """The run's directory is fetched: the runner's agent removes it."""
        _ssh(self.remote, self.remote.command(self._argv("fetched"))[-1])


def _json_line(output: str) -> dict:
    return json.loads(output.strip().splitlines()[-1])

