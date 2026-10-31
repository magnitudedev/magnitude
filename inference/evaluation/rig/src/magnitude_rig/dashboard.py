"""`mrig dashboard`: a local web page showing every host live: what it is, which runs hold it
and which wait, its utilization, memory and temperatures, and every run on any host, whichever
machine launched it.

Each host keeps one `watch` stream (`python -m magnitude_rig.host watch`) open over ssh,
feeding the latest view kept here; the page polls that view. A host that cannot be reached,
or whose stream ends, is shown so and reconnected after RETRY_SECONDS. Nothing is held on any
host and nothing is written there but the host package `connect` deploys.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .connection import HostUnavailable, connect
from .hosts import HostConfig, Hosts

PAGE = (Path(__file__).parent / "dashboard.html").read_bytes()
RETRY_SECONDS = 15.0
STDERR_LINES = 20


class Watch:
    """One host's `watch` stream, reconnected until stopped, and the latest it reported."""

    def __init__(self, config: HostConfig, interval: float, stopping: threading.Event):
        self._config = config
        self._interval = interval
        self._stopping = stopping
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._view: dict = {"alias": config.alias, "state": "connecting"}
        self._thread = threading.Thread(target=self._run, name=f"watch-{config.alias}", daemon=True)
        self._thread.start()

    def view(self) -> dict:
        with self._lock:
            return dict(self._view)

    def _set(self, **fields) -> None:
        with self._lock:
            self._view.update(fields)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                self._stream()
            except HostUnavailable as error:
                if not self._stopping.is_set():
                    self._set(state="unreachable", error=str(error))
            self._stopping.wait(RETRY_SECONDS)

    def _stream(self) -> None:
        remote = connect(self._config)
        self._set(
            state="connecting", error=None, backends=list(remote.backends), ready=remote.ready,
            interval=self._interval,
        )
        argv = remote.host(
            "magnitude_rig.host", "watch", "--root", remote.root, "--alias", remote.alias,
            "--interval", str(self._interval),
        )
        with self._lock:
            if self._stopping.is_set():
                return
            self._process = subprocess.Popen(
                remote.command(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            process = self._process
        stderr: deque[str] = deque(maxlen=STDERR_LINES)
        drain = threading.Thread(target=lambda: stderr.extend(process.stderr), daemon=True)
        drain.start()
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue  # the remote shell's startup output, before the stream begins
            if "facts" in event:
                self._set(facts=event["facts"])
            else:
                self._set(state="live", sample=event, received=time.time())
        code = process.wait()
        drain.join()
        raise HostUnavailable(f"watch exited {code}: {''.join(stderr).strip()}")

    def close(self) -> None:
        """Begin ending the stream (call with `stopping` set): closing its stdin ends `watch`
        on the host."""
        with self._lock:
            if self._process is not None:
                self._process.stdin.close()

    def join(self) -> None:
        """Wait for the stream to end after `close`, killing it if it does not."""
        with self._lock:
            process = self._process
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._thread.join(timeout=10)


def _handler(snapshot):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send("text/html; charset=utf-8", PAGE)
            elif self.path == "/api/snapshot":
                self._send("application/json", json.dumps(snapshot()).encode())
            else:
                self.send_error(404)

        def _send(self, kind: str, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *arguments) -> None:
            pass  # one request per poll; the page is the output

    return Handler


def serve(arguments) -> int:
    hosts = Hosts.load()
    configs = (
        tuple(dict.fromkeys(c for name in arguments.host for c in hosts.resolve(name)))
        if arguments.host
        else tuple(hosts.host(alias) for alias in hosts.known())
    )
    if not configs:
        raise SystemExit("no hosts: list them in ~/.magnitude/evaluation/hosts.json or pass --host")

    def snapshot() -> dict:
        return {
            "now": time.time(),
            "pools": hosts.pools,
            "hosts": [watch.view() for watch in watches],
        }

    try:
        server = ThreadingHTTPServer(("127.0.0.1", arguments.port), _handler(snapshot))
    except OSError as error:
        raise SystemExit(f"cannot serve on port {arguments.port}: {error}; pass another --port")
    stopping = threading.Event()
    watches = [Watch(config, arguments.interval, stopping) for config in configs]
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"mrig dashboard: {url} ({len(watches)} hosts); Ctrl-C to stop", flush=True)
    if not arguments.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", file=sys.stderr)
    finally:
        stopping.set()
        server.server_close()
        for watch in watches:
            watch.close()
        for watch in watches:
            watch.join()
    return 0


def configure(command) -> None:
    command.add_argument("--host", action="append", default=[],
                         help="a host or pool to show (repeatable); every listed host by default")
    command.add_argument("--port", type=int, default=8765, help="local port (default 8765)")
    command.add_argument("--interval", type=float, default=2.0,
                         help="seconds between each host's readings (default 2)")
    command.add_argument("--no-open", action="store_true", help="do not open a browser")
    command.set_defaults(handler=serve)
