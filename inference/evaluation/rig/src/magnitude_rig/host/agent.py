"""The rig's host agent: one session per run, taking and releasing typed holds on the host.

Copied verbatim to `<root>/agent/<sha256 prefix>.py` and run with the host's system Python,
so it is a single standard-library file (Python 3.9+) importing nothing from this package.

    agent.py session --root R --who W --run ID --purpose P --estimate S [--heartbeat-timeout S]
    agent.py status  --root R [--checkout KEY]
    agent.py prune   --dir D   (manifest of relative paths to keep on stdin)

A hold is what a phase of a run needs from the host. Kinds, and which conflict:

    build         compiling             conflicts with timed
    device        running on the GPU    conflicts with device, timed
    timed         measuring             conflicts with build, device, timed
    models        the shared model store, while acquiring into it
    checkout:KEY  using one checkout (its source, build target and caches): conflicts with
                  nothing, and eviction never removes a checkout with a live hold. A key is
                  its content, so every run of one key needs the same tree
    mirror:KEY    writing one checkout's source, while mirroring and pruning it

Kinds conflict only as listed (each of `models` and `mirror:KEY` with itself); holds of one
session never conflict with each other. Requests are served in arrival order: a request is
granted when no live hold conflicts with it and no live earlier request does. A session takes
its holds in the order checkout, mirror (released before any other), build, models, device or
timed, and requests device or timed only while holding no build or models hold, so no two
sessions ever wait on each other.

Every request and hold is an entry `<root>/rig/entries/<sequence>-<pid>-<n>.json`, which its
session keeps open under an exclusive OS lock for as long as it lives: an entry whose lock can
be taken belongs to a dead session and is removed by whoever looks next, so nothing outlives
its session. Entries are created, changed and judged only under `<root>/rig/guard.lock`. A
device or timed hold also takes `<root>/lock`, the device lock other tools on the host take.

`session` speaks JSON Lines: requests on stdin, events on stdout. Any request counts as a
heartbeat.

    -> {"op": "heartbeat"}
    -> {"op": "hold", "kind": K}
    -> {"op": "release", "kind": K}
    -> {"op": "run", "argv": [...], "cwd": "...", "env": {...}, "input": "..."}
    -> {"op": "evict", "keep": KEY}
    -> {"op": "kill"}              (end the running command, if any)
    -> {"op": "end"}
    <- {"event": "waiting", "kind": K, "blocking": [entry, ...]}
    <- {"event": "held", "kind": K}
    <- {"event": "released", "kind": K}
    <- {"event": "output", "stream": "stdout|stderr", "text": "..."}
    <- {"event": "exited", "code": N}
    <- {"event": "evicted", "removed": [KEY, ...], "free_bytes": N}
    <- {"event": "killed"}
    <- {"event": "error", "message": "..."}
    <- {"event": "stopped", "reason": "..."}

On stdin EOF, heartbeat loss or a termination signal the agent kills its child's whole process
group, withdraws its entries and exits. After each command exits, survivors in its process
group are killed too. Nothing else ends a session: no hold has a time limit.

Each checkout lives in `<root>/checkouts/<KEY>/`; `used` is touched whenever it is held, and
`size` records its build target's size after a build from cold. `evict` removes the
least recently used checkouts no live session holds until the volume has as much free space as
the largest recorded target, so a build from cold always fits.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

POLL_SECONDS = 0.5
TERMINATE_GRACE_SECONDS = 10.0
_WINDOWS = os.name == "nt"
CONFLICTS = {
    ("build", "timed"),
    ("device", "device"),
    ("device", "timed"),
    ("timed", "timed"),
    ("models", "models"),
}
DEVICE_KINDS = ("device", "timed")
ENTRY_LOCK_OFFSET = 1 << 30
# The entries' meaning. An entry of another protocol is taken to conflict with every request,
# so checkouts on different commits never run what this one's rules would keep apart.
PROTOCOL = 2


def conflicts(a: str, b: str) -> bool:
    if a.startswith("mirror:") or b.startswith("mirror:"):
        return a == b
    if a.startswith("checkout:") or b.startswith("checkout:"):
        return False
    return (a, b) in CONFLICTS or (b, a) in CONFLICTS


class FileLock:
    """A non-blocking OS lock on a file; closing it (or exiting) releases it.

    Windows locks a byte range and refuses reads of it, so an entry, whose content is read
    while it is locked, locks a byte far past its content (`offset`); the device lock keeps
    byte 0, where other tools on the host lock it."""

    def __init__(self, path: str, mode: str = "a+b", offset: int = 0):
        self.path = path
        self.offset = offset
        self.file = open(path, mode)  # not inherited by children (PEP 446)

    def try_acquire(self, exclusive: bool = True) -> bool:
        try:
            if _WINDOWS:
                import msvcrt

                self.file.seek(self.offset)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(self.file.fileno(), flag | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def close(self) -> None:
        self.file.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Rig:
    """The host's rig directory: entries and the guard serializing every change to them."""

    def __init__(self, root: str):
        self.root = root
        self.directory = os.path.join(root, "rig")
        self.entries = os.path.join(self.directory, "entries")
        os.makedirs(self.entries, exist_ok=True)

    def guarded(self, action):
        guard = FileLock(os.path.join(self.directory, "guard.lock"))
        try:
            while not guard.try_acquire():
                time.sleep(0.01)
            return action()
        finally:
            guard.close()

    def live(self) -> list:
        """Live entries in arrival order, removing those whose session is gone. Call guarded."""
        found = []
        for name in sorted(os.listdir(self.entries)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.entries, name)
            try:
                probe = FileLock(path, "r+b", ENTRY_LOCK_OFFSET)
            except OSError:
                continue
            try:
                if probe.try_acquire(exclusive=False):
                    probe.close()
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    continue
                probe.file.seek(0)
                entry = json.loads(probe.file.read().decode())
            except ValueError:
                continue
            finally:
                probe.close()
            entry["entry"] = name
            found.append(entry)
        return found

    def checkout(self, key: str) -> str:
        return os.path.join(self.root, "checkouts", key)


def remaining(entry: dict, now: float) -> float:
    """Seconds left of the entry's run by its own estimate (never negative)."""
    return max(0.0, entry["estimate"] - (now - entry["started"]))


class Session:
    def __init__(self, rig: Rig, info: dict):
        self.rig = rig
        self.info = info
        self.entries = {}  # kind -> (path, lock)
        self.held = set()
        self.device = None
        self._count = 0

    def _write(self, kind: str, state: str) -> None:
        path, lock = self.entries[kind]
        entry = dict(self.info, kind=kind, state=state, changed=_now())
        lock.file.seek(0)
        lock.file.truncate()
        lock.file.write(json.dumps(entry).encode())
        lock.file.flush()

    def request(self, kind: str) -> None:
        def enter():
            sequence = 1 + max((int(e["entry"].split("-")[0]) for e in self.rig.live()), default=0)
            self._count += 1
            name = "%012d-%d-%d.json" % (sequence, os.getpid(), self._count)
            path = os.path.join(self.rig.entries, name)
            lock = FileLock(path, "w+b", ENTRY_LOCK_OFFSET)
            if not lock.try_acquire():  # a new file under the guard: nothing else can hold it
                raise OSError("cannot lock the new entry %s" % path)
            self.entries[kind] = (path, lock)
            self._write(kind, "waiting")

        self.rig.guarded(enter)

    def blocking(self, kind: str) -> list:
        """Entries that keep `kind` from being held now; empty once it is held."""

        def judge():
            mine = os.path.basename(self.entries[kind][0])
            ahead = True
            blockers = []
            for entry in self.rig.live():
                if entry["entry"] == mine:
                    ahead = False
                    continue
                if entry["run"] == self.info["run"]:
                    continue
                if entry.get("protocol") == PROTOCOL and not conflicts(entry["kind"], kind):
                    continue
                if entry["state"] == "held" or ahead:
                    blockers.append(entry)
            if blockers:
                return blockers
            if kind in DEVICE_KINDS and self.device is None:
                device = FileLock(os.path.join(self.rig.root, "lock"))
                if not device.try_acquire():
                    device.close()
                    return [{"kind": "device", "who": "a tool holding %s" % device.path,
                             "run": None, "state": "held"}]
                self.device = device
            self._write(kind, "held")
            self.held.add(kind)
            if kind.startswith("checkout:"):
                directory = self.rig.checkout(kind.split(":", 1)[1])
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, "used"), "w") as stream:
                    stream.write(_now())
            return []

        return self.rig.guarded(judge)

    def release(self, kind: str) -> None:
        def withdraw():
            entry = self.entries.pop(kind, None)
            if entry is None:
                return
            path, lock = entry
            try:
                os.remove(path)
            except OSError:
                pass
            lock.close()
            self.held.discard(kind)
            if self.device is not None and not self.held.intersection(DEVICE_KINDS):
                self.device.close()
                self.device = None

        self.rig.guarded(withdraw)

    def release_all(self) -> None:
        for kind in list(self.entries):
            self.release(kind)

    def evict(self, keep: str) -> dict:
        def act():
            held = {e["kind"].split(":", 1)[1] for e in self.rig.live() if e["kind"].startswith("checkout:")}
            base = os.path.join(self.rig.root, "checkouts")
            known = []
            for key in os.listdir(base) if os.path.isdir(base) else ():
                directory = os.path.join(base, key)
                known.append((_mtime(os.path.join(directory, "used")), key, _size(directory)))
            needed = max((size for _, _, size in known), default=0)
            removed = []
            for _, key, _ in sorted(known):
                if shutil.disk_usage(self.rig.root).free >= needed:
                    break
                if key in held or key == keep:
                    continue
                shutil.rmtree(os.path.join(base, key), ignore_errors=True)
                removed.append(key)
            return {"removed": removed, "free_bytes": shutil.disk_usage(self.rig.root).free}

        return self.rig.guarded(act)


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _size(directory: str) -> int:
    try:
        with open(os.path.join(directory, "size")) as stream:
            return int(stream.read().strip())
    except (OSError, ValueError):
        return 0


class Output:
    """Serialized JSON Lines events; a vanished controller is handled by the heartbeat."""

    def __init__(self):
        self._lock = threading.Lock()

    def emit(self, event: str, **fields) -> None:
        line = json.dumps({"event": event, **fields}) + "\n"
        with self._lock:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass


def kill_group(process: subprocess.Popen, grace: float) -> None:
    """Terminate the child's whole process group, escalating to a kill after `grace`."""
    if _WINDOWS:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Child:
    """One command in its own process group, with its output streamed as events."""

    def __init__(self, request: dict, output: Output):
        env = dict(os.environ)
        env.update(request.get("env", {}))
        data = request.get("input")
        options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if _WINDOWS
            else {"start_new_session": True}
        )
        self.process = subprocess.Popen(
            request["argv"],
            cwd=request.get("cwd"),
            env=env,
            stdin=subprocess.DEVNULL if data is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **options,
        )
        self._output = output
        self._pumps = [
            threading.Thread(target=self._pump, args=(self.process.stdout, "stdout"), daemon=True),
            threading.Thread(target=self._pump, args=(self.process.stderr, "stderr"), daemon=True),
        ]
        for pump in self._pumps:
            pump.start()
        if data is not None:
            threading.Thread(target=self._feed, args=(data,), daemon=True).start()
        self.finished = threading.Event()
        threading.Thread(target=self._wait, daemon=True).start()

    def _feed(self, data: str) -> None:
        try:
            self.process.stdin.write(data.encode())
            self.process.stdin.close()
        except OSError:
            pass

    def _pump(self, stream, name: str) -> None:
        for line in iter(stream.readline, b""):
            self._output.emit("output", stream=name, text=line.decode("utf-8", "replace"))

    def _wait(self) -> None:
        code = self.process.wait()
        if not _WINDOWS:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for pump in self._pumps:
            pump.join()
        self._output.emit("exited", code=code)
        self.finished.set()

    def kill(self) -> None:
        kill_group(self.process, TERMINATE_GRACE_SECONDS)
        self.finished.wait(TERMINATE_GRACE_SECONDS)


def _requests(inbox: "queue.Queue") -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            inbox.put(json.loads(line))
        except ValueError:
            inbox.put({"op": "invalid", "line": line})
    inbox.put(None)


def user_tool_path() -> None:
    """Put the user's tool directories (uv, cargo) first on PATH for every child."""
    home = os.path.expanduser("~")
    tools = [os.path.join(home, ".local", "bin"), os.path.join(home, ".cargo", "bin")]
    os.environ["PATH"] = os.pathsep.join(tools + [os.environ.get("PATH", "")])


def _public(entry: dict) -> dict:
    return {k: entry[k] for k in ("kind", "who", "run", "purpose", "estimate", "started", "state") if k in entry}


def session(arguments) -> int:
    user_tool_path()
    rig = Rig(arguments.root)
    output = Output()
    signalled = []
    for name in ("SIGTERM", "SIGHUP", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda signum, frame: signalled.append(signum))
    inbox: "queue.Queue" = queue.Queue()
    threading.Thread(target=_requests, args=(inbox,), daemon=True).start()
    current = Session(
        rig,
        {
            "who": arguments.who,
            "run": arguments.run,
            "purpose": arguments.purpose,
            "estimate": arguments.estimate,
            "started": time.time(),
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        },
    )
    heartbeat = time.monotonic()
    pending = None  # (kind, last reported blockers)
    child = None
    reason = None
    while True:
        now = time.monotonic()
        if signalled:
            reason = "signal %s" % signalled[0]
        elif now - heartbeat > arguments.heartbeat_timeout:
            reason = "no heartbeat for %.0f s" % (now - heartbeat)
        elif pending is not None:
            kind, reported = pending
            blockers = [_public(e) for e in current.blocking(kind)]
            if not blockers:
                output.emit("held", kind=kind)
                pending = None
            elif [b.get("run") for b in blockers] != reported:
                output.emit("waiting", kind=kind, blocking=blockers)
                pending = (kind, [b.get("run") for b in blockers])
        if reason is not None:
            break
        try:
            request = inbox.get(timeout=POLL_SECONDS)
        except queue.Empty:
            continue
        if request is None:
            reason = "controller closed the session"
            break
        heartbeat = time.monotonic()
        op = request.get("op")
        if op == "end":
            reason = "ended"
            break
        if op == "hold":
            kind = request["kind"]
            if pending is not None or kind in current.entries:
                output.emit("error", message="hold %s while %s is pending or held" % (kind, pending))
            elif kind in DEVICE_KINDS and current.held.intersection(("build", "models")):
                output.emit("error", message="hold %s while holding %s" % (kind, sorted(current.held)))
            else:
                current.request(kind)
                pending = (kind, None)
        elif op == "release":
            current.release(request["kind"])
            output.emit("released", kind=request["kind"])
        elif op == "evict":
            output.emit("evicted", **current.evict(request["keep"]))
        elif op == "kill":
            if child is not None and not child.finished.is_set():
                child.kill()
            output.emit("killed")
        elif op == "run":
            if child is not None and not child.finished.is_set():
                output.emit("error", message="a command is already running")
            else:
                try:
                    child = Child(request, output)
                except OSError as error:
                    output.emit("output", stream="stderr", text="agent: %s\n" % error)
                    output.emit("exited", code=127)
        elif op != "heartbeat":
            output.emit("error", message="unknown request %s" % request)
    if child is not None and not child.finished.is_set():
        child.kill()
    current.release_all()
    output.emit("stopped", reason=reason)
    return 0 if reason == "ended" else 2


def status(arguments) -> int:
    """The host's live entries, the queue a new device or timed request would wait behind (by
    the entries' own estimates), and whether a checkout is present on the host."""
    rig = Rig(arguments.root)
    entries = rig.guarded(rig.live)
    now = time.time()
    waits = {}
    for kind in ("build", "device", "timed"):
        waits[kind] = sum(
            remaining(e, now) for e in entries if e.get("protocol") != PROTOCOL or conflicts(e["kind"], kind)
        )
    legacy = None
    device = FileLock(os.path.join(arguments.root, "lock"))
    if not any(e["kind"] in DEVICE_KINDS and e["state"] == "held" for e in entries):
        if not device.try_acquire():
            try:
                with open(os.path.join(arguments.root, "holder.json")) as stream:
                    legacy = json.load(stream)
            except (OSError, ValueError):
                legacy = {}
    device.close()
    report = {
        "entries": [dict(_public(e), remaining=remaining(e, now)) for e in entries],
        "waits": waits,
        "device_lock_held_elsewhere": legacy,
        "free_bytes": shutil.disk_usage(arguments.root).free,
    }
    if arguments.checkout:
        directory = rig.checkout(arguments.checkout)
        report["checkout"] = {
            "present": os.path.isdir(os.path.join(directory, "target")),
            "size": _size(directory),
        }
    print(json.dumps(report))
    return 0


def prune(arguments) -> int:
    """Delete files and directories under --dir that the stdin manifest does not list."""
    keep = {os.path.normpath(line.strip()) for line in sys.stdin if line.strip()}
    parents = set()
    for path in keep:
        head = os.path.dirname(path)
        while head and head not in parents:
            parents.add(head)
            head = os.path.dirname(head)
    removed = 0
    for current, directories, files in os.walk(arguments.dir, topdown=False):
        relative_dir = os.path.relpath(current, arguments.dir)
        links = [d for d in directories if os.path.islink(os.path.join(current, d))]
        for name in files + links:
            relative = os.path.normpath(os.path.join(relative_dir, name))
            if relative not in keep:
                os.remove(os.path.join(current, name))
                removed += 1
        for name in directories:
            path = os.path.join(current, name)
            relative = os.path.normpath(os.path.join(relative_dir, name))
            if name not in links and relative not in parents and not os.listdir(path):
                os.rmdir(path)
    print(json.dumps({"removed_files": removed}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="magnitude-rig-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("session")
    command.add_argument("--root", required=True)
    command.add_argument("--who", required=True)
    command.add_argument("--run", required=True)
    command.add_argument("--purpose", required=True)
    command.add_argument("--estimate", type=float, required=True)
    command.add_argument("--heartbeat-timeout", type=float, default=90.0)
    command.set_defaults(handler=session)
    command = commands.add_parser("status")
    command.add_argument("--root", required=True)
    command.add_argument("--checkout")
    command.set_defaults(handler=status)
    command = commands.add_parser("prune")
    command.add_argument("--dir", required=True)
    command.set_defaults(handler=prune)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
