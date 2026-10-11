"""The evaluation host agent: one lock holder per controlling session.

Copied verbatim to `<root>/agent/<sha256 prefix>.py` and run with the host's system Python,
so it is a single standard-library file (Python 3.9+) importing nothing from this package.

    agent.py hold   --root R --who W --run ID --max-hold S [--heartbeat-timeout S]
    agent.py status --root R
    agent.py prune  --dir D   (manifest of relative paths to keep on stdin)

`hold` queues for an OS file lock on `<root>/lock` (flock / msvcrt.locking) in arrival
order (see `Queue`), publishes `<root>/holder.json`, then runs the controller's commands
one at a time as children in their own process group, with `~/.local/bin` and
`~/.cargo/bin` first on PATH. It speaks JSON Lines: requests on stdin, events on stdout.

    -> {"op": "heartbeat"}
    -> {"op": "run", "argv": [...], "cwd": "...", "env": {...}, "input": "..."}
    -> {"op": "release"}
    <- {"event": "waiting", "holder": {...}, "ahead": N}
    <- {"event": "locked", "holder": {...}}
    <- {"event": "output", "stream": "stdout|stderr", "text": "..."}
    <- {"event": "exited", "code": N}
    <- {"event": "stopped", "reason": "..."}

Any request counts as a heartbeat. On stdin EOF, heartbeat loss, the hard maximum hold
time or a termination signal the agent kills its child's whole process group, withdraws
`holder.json` while still holding the lock, and exits; the kernel releases the lock with
the process, so a lock never outlives its work and work never outlives its lock. After
each command exits, survivors in its process group are killed too.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

POLL_SECONDS = 1.0
TICKET_STALE_SECONDS = 30.0
TERMINATE_GRACE_SECONDS = 10.0
_WINDOWS = os.name == "nt"


class FileLock:
    """An exclusive, non-blocking OS lock; closing the file (or exiting) releases it."""

    def __init__(self, path: str):
        self._file = open(path, "a+b")  # not inherited by children (PEP 446)

    def try_acquire(self) -> bool:
        try:
            if _WINDOWS:
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def close(self) -> None:
        self._file.close()


def _guarded(root: str, action):
    """Run `action` under the short-held queue guard lock."""
    guard = FileLock(os.path.join(root, "queue.lock"))
    try:
        while not guard.try_acquire():
            time.sleep(0.01)
        return action()
    finally:
        guard.close()


def _pid_alive(pid: int) -> bool:
    if _WINDOWS:  # os.kill would terminate the process; rely on the ticket's freshness
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def queue_tickets(root: str, now: float):
    """Tickets in arrival order as (path, info, live)."""
    directory = os.path.join(root, "queue")
    tickets = []
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else ():
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            fresh = now - os.path.getmtime(path) <= TICKET_STALE_SECONDS
            with open(path) as stream:
                info = json.load(stream)
        except (OSError, ValueError):
            continue  # withdrawn while listing, or still being renamed into place
        tickets.append((path, info, fresh and _pid_alive(info["pid"])))
    return tickets


class Queue:
    """FIFO admission to the main lock: only the earliest live ticket may try to take it.

    A ticket is `<root>/queue/<sequence>-<pid>.json`, created under the guard lock with a
    sequence above every existing ticket. Its owner touches it every poll; a ticket whose
    process is gone or whose file is stale is removed by whoever looks next, so a crashed
    waiter never blocks the queue. A waiter whose ticket was removed (it stalled) re-queues.
    """

    def __init__(self, root: str, info: dict):
        self.root = root
        self.info = info
        self.path = None
        os.makedirs(os.path.join(root, "queue"), exist_ok=True)

    def _enter(self) -> None:
        tickets = queue_tickets(self.root, time.time())
        sequence = 1 + max((int(os.path.basename(p).split("-")[0]) for p, _, _ in tickets), default=0)
        path = os.path.join(self.root, "queue", "%012d-%d.json" % (sequence, os.getpid()))
        info = dict(self.info, queued_since=_now())
        with open(path + ".tmp", "w") as stream:
            json.dump(info, stream)
        os.replace(path + ".tmp", path)
        self.path = path

    def ahead(self) -> int:
        """Live tickets ahead of ours (queuing first, or again if ours was removed)."""

        def look() -> int:
            if self.path is None or not os.path.exists(self.path):
                self._enter()
            os.utime(self.path)
            position = 0
            for path, _, live in queue_tickets(self.root, time.time()):
                if path == self.path:
                    return position
                if live:
                    position += 1
                else:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            raise OSError("own queue ticket vanished")

        return _guarded(self.root, look)

    def leave(self) -> None:
        if self.path is not None:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.path = None


def _read_holder(path: str):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Output:
    """Serialized JSON Lines events; a vanished controller is handled by the watchdog."""

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


def hold(arguments) -> int:
    user_tool_path()
    root = arguments.root
    os.makedirs(root, exist_ok=True)
    holder_path = os.path.join(root, "holder.json")
    lock = FileLock(os.path.join(root, "lock"))
    output = Output()
    signalled = []
    for name in ("SIGTERM", "SIGHUP", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda signum, frame: signalled.append(signum))
    inbox: "queue.Queue" = queue.Queue()
    threading.Thread(target=_requests, args=(inbox,), daemon=True).start()
    holder = {
        "who": arguments.who,
        "run": arguments.run,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "max_hold_seconds": arguments.max_hold,
    }
    waiting = Queue(root, holder)
    heartbeat = time.monotonic()
    locked_at = None
    child = None
    seen_holder = object()
    reason = None
    while True:
        now = time.monotonic()
        if signalled:
            reason = f"signal {signalled[0]}"
        elif now - heartbeat > arguments.heartbeat_timeout:
            reason = f"no heartbeat for {now - heartbeat:.0f} s"
        elif locked_at is not None and now - locked_at > arguments.max_hold:
            reason = f"maximum hold time {arguments.max_hold:.0f} s reached"
        elif locked_at is None:
            ahead = waiting.ahead()
            if ahead == 0 and lock.try_acquire():
                locked_at = now
                holder["since"] = _now()
                with open(holder_path + ".tmp", "w") as stream:
                    json.dump(holder, stream)
                os.replace(holder_path + ".tmp", holder_path)
                waiting.leave()
                output.emit("locked", holder=holder)
            else:
                current = (_read_holder(holder_path), ahead)
                if current != seen_holder:
                    seen_holder = current
                    output.emit("waiting", holder=current[0], ahead=ahead)
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
        if op == "release":
            reason = "released"
            break
        if op == "run":
            if locked_at is None:
                output.emit("error", message="run before the lock was acquired")
            elif child is not None and not child.finished.is_set():
                output.emit("error", message="a command is already running")
            else:
                try:
                    child = Child(request, output)
                except OSError as error:
                    output.emit("output", stream="stderr", text=f"agent: {error}\n")
                    output.emit("exited", code=127)
        elif op != "heartbeat":
            output.emit("error", message=f"unknown request {request}")
    if child is not None and not child.finished.is_set():
        child.kill()
    if locked_at is not None:
        try:
            os.remove(holder_path)
        except OSError:
            pass
    waiting.leave()
    lock.close()
    output.emit("stopped", reason=reason)
    return 0 if reason == "released" else 2


def status(arguments) -> int:
    path = os.path.join(arguments.root, "lock")
    holder = _read_holder(os.path.join(arguments.root, "holder.json"))
    if not os.path.exists(path):
        locked = False
    else:
        probe = FileLock(path)
        locked = not probe.try_acquire()
        probe.close()
    waiters = [
        dict(info, live=live) for _, info, live in queue_tickets(arguments.root, time.time())
    ]
    print(json.dumps({"root": arguments.root, "locked": locked, "holder": holder if locked else None,
                      "stale_holder": None if locked else holder, "queue": waiters}))
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
    parser = argparse.ArgumentParser(prog="magnitude-evaluation-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("hold")
    command.add_argument("--root", required=True)
    command.add_argument("--who", required=True)
    command.add_argument("--run", required=True)
    command.add_argument("--max-hold", type=float, required=True)
    command.add_argument("--heartbeat-timeout", type=float, default=90.0)
    command.set_defaults(handler=hold)
    command = commands.add_parser("status")
    command.add_argument("--root", required=True)
    command.set_defaults(handler=status)
    command = commands.add_parser("prune")
    command.add_argument("--dir", required=True)
    command.set_defaults(handler=prune)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
