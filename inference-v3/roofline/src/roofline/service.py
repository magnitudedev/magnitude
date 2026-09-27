"""Durable coordinator and worker supervisors, with independent numerical owners."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import json
import os
import select
import signal
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from .contracts import Attempt, Measurement, Request, Source, encoded, now
from .store import Store, atomic_write
from .transport import (
    WorkerClient,
    WorkerError,
    ensure_service,
    failure,
    local_call,
    receive,
    send,
    socket_path,
)

TERMINAL = {"complete", "failed", "cancelled", "unavailable"}


def transfer_source(client, store, source):
    def deliver(message):
        while True:
            try:
                client.call(message)
                return
            except RuntimeError as exc:
                if str(exc) != "worker busy":
                    raise
                time.sleep(0.5)

    missing = client.call({"op": "missing", "blobs": list({f.blob for f in source.files})})
    batch, size = {}, 0
    for blob in missing:
        content = store.blob(blob)
        if len(content) <= 1 << 20:
            if size + len(content) > 1 << 20:
                deliver({"op": "blobs", "contents": batch})
                batch, size = {}, 0
            batch[blob] = base64.b64encode(content).decode()
            size += len(content)
        else:
            for offset in range(0, len(content), 1 << 20):
                chunk = content[offset : offset + (1 << 20)]
                deliver(
                    {
                        "op": "blob",
                        "id": blob,
                        "offset": offset,
                        "content": base64.b64encode(chunk).decode(),
                        "final": offset + len(chunk) == len(content),
                    }
                )
    if batch:
        deliver({"op": "blobs", "contents": batch})
    client.call({"op": "source", "source": source.model_dump()})


def collect_blob(client, store, blob):
    if store.blob_path(blob).exists():
        store.blob(blob)
        return
    content = bytearray()
    while True:
        page = client.call({"op": "get_blob", "id": blob, "offset": len(content)})
        content.extend(base64.b64decode(page["content"]))
        if page["complete"]:
            break
    if store.put_blob(bytes(content)) != blob:
        raise ValueError("worker artifact checksum mismatch")


class Service:
    def __init__(self, root, project, kind):
        self.root, self.project, self.kind = root, project, kind
        self.lock = threading.RLock()
        self.running = set()
        self.executor = None
        self.executor_source = None
        self.executor_directory = None
        self.refreshed = False
        self.active = None
        self.shutdown_requested = False
        self.cancel_requested = threading.Event()
        self.paused = (root / "paused").exists()
        self.last_used = time.monotonic()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def dispatch(self, message):
        op = message["op"]
        if op == "discover":
            return self.discover(message)
        with self.lock, Store(self.root) as store:
            if op == "preflight":
                from .discovery import preflight

                return preflight(self.root, store, message)
            if op == "shutdown":
                if self.active or any(
                    r["status"] in {"accepted", "running"} for r in store.records("attempt")
                ):
                    raise RuntimeError("worker busy; setup cannot replace it during accepted work")
                self.shutdown_requested = True
                return True
            if op == "ping":
                return {
                    "version": 1,
                    "root": str(self.root),
                    "python": sys.executable,
                    "kind": self.kind,
                    "paused": self.paused,
                    "active": self.active,
                    "pid": os.getpid(),
                    "updated": now(),
                }
            if op in ("pause", "resume"):
                self.paused = op == "pause"
                if self.paused:
                    atomic_write(self.root / "paused", b"paused")
                else:
                    (self.root / "paused").unlink(missing_ok=True)
                return {"paused": self.paused}
            if op == "missing":
                return [b for b in message["blobs"] if not store.blob_path(b).exists()]
            if op == "blobs":
                if self.active:
                    raise RuntimeError("worker busy")
                for blob, content in message["contents"].items():
                    store.blob_path(blob)
                    if store.put_blob(base64.b64decode(content, validate=True)) != blob:
                        raise ValueError("received blob checksum mismatch")
                return True
            if op == "blob":
                if self.active:
                    raise RuntimeError("worker busy")
                blob = message["id"]
                store.blob_path(blob)  # Validate identity before constructing any path.
                temporary = self.root / "incoming" / blob
                temporary.parent.mkdir(parents=True, exist_ok=True)
                offset = message["offset"]
                if not isinstance(offset, int) or offset < 0:
                    raise ValueError("invalid blob offset")
                if offset == 0:
                    temporary.write_bytes(b"")
                if not temporary.exists() or temporary.stat().st_size != offset:
                    raise ValueError("noncontiguous blob delivery")
                with temporary.open("ab") as stream:
                    stream.write(base64.b64decode(message["content"], validate=True))
                if message["final"]:
                    if store.put_blob(temporary.read_bytes()) != blob:
                        raise ValueError("received blob checksum mismatch")
                    temporary.unlink()
                return True
            if op == "get_blob":
                if self.active:
                    raise RuntimeError("worker busy")
                content = store.blob(message["id"])
                start = int(message.get("offset", 0))
                if start < 0 or start > len(content):
                    raise ValueError("invalid artifact offset")
                part = content[start : start + (1 << 20)]
                return {
                    "content": base64.b64encode(part).decode(),
                    "complete": start + len(part) == len(content),
                }
            if op == "source":
                source = Source.model_validate(message["source"])
                for item in source.files:
                    if not store.blob_path(item.blob).exists():
                        raise ValueError("source references an undelivered blob")
                store.put("source", source.source_id, source)
                return source.source_id
            if op == "accept":
                request = Request.model_validate(message["request"])
                attempt = Attempt.model_validate(message["attempt"])
                if self.paused:
                    raise RuntimeError("worker paused")
                if attempt.attempt_id not in {a.attempt_id for a in request.attempts}:
                    raise ValueError("attempt is not part of request")
                store.source(request.experiment.source)
                for blob in request.inputs.values():
                    store.blob(blob)
                try:
                    previous = store.get("attempt", attempt.attempt_id)
                    if {
                        k: v
                        for k, v in previous["request"].items()
                        if k not in ("attempts", "cancelled")
                    } != request.model_dump(mode="json", exclude={"attempts", "cancelled"}):
                        raise ValueError("attempt identity collision")
                    return previous["status"]
                except KeyError:
                    record = {
                        "request": request.model_dump(mode="json"),
                        "attempt": attempt.model_dump(mode="json"),
                        "status": "accepted",
                        "measurements": [],
                        "artifacts": [],
                        "error": None,
                    }
                    store.put("attempt", attempt.attempt_id, record)
                    return "accepted"
            if op == "status":
                return store.get("attempt", message["attempt_id"])
            if op == "cancel":
                if self.kind == "coordinator":
                    request = store.request(message["request_id"])
                    store.put(
                        "request",
                        request.request_id,
                        request.model_copy(update={"cancelled": True}),
                        mutable=True,
                    )
                else:
                    record = store.get("attempt", message["attempt_id"])
                    if record["status"] not in TERMINAL:
                        record["cancelled"] = True
                        store.put("attempt", message["attempt_id"], record, mutable=True)
                        if self.active == message["attempt_id"]:
                            self.cancel_requested.set()
                            self.stop_executor()
                return True
            if op == "wake":
                return True
            raise ValueError(f"unknown control operation: {op}")

    def stop_executor(self):
        process, self.executor = self.executor, None
        self.executor_source = None
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def discover(self, message):
        import json

        with self.lock:
            if self.active:
                raise RuntimeError("worker busy")
            self.active = "scope discovery"
        try:
            from .environment import available_environment, execution_paths, worker_environment
            from .sources import materialize

            with Store(self.root) as store:
                prepared = available_environment(self.root, message["dependency_id"], store)
                if prepared is None:
                    raise ValueError("metadata discovery requires a prepared matching environment")
                source = store.source(message["source_id"])
                if any(not f.path.startswith(("src/", "roofline/src/")) for f in source.files):
                    raise ValueError("metadata discovery accepts only engine Python source")
                destination = materialize(source, store, self.root / "discovery" / source.source_id)
                destination, python, env = execution_paths(
                    destination, prepared, worker_environment(self.root)
                )
            result = subprocess.run(
                [str(python), "-S", "-m", "roofline.discovery"],
                input=json.dumps(message["payload"]),
                capture_output=True,
                text=True,
                cwd=destination,
                env=env,
                timeout=120,
            )
            if result.returncode:
                raise RuntimeError(result.stderr[-8000:])
            return json.loads(result.stdout)
        finally:
            with self.lock:
                self.active = None

    def own_process(self, process):
        self.executor = process
        started = subprocess.check_output(
            ["ps", "-p", str(process.pid), "-o", "lstart="], text=True
        ).strip()
        atomic_write(
            self.root / "executor-owner.json", encoded({"pid": process.pid, "started": started})
        )
        return process

    def reap_previous_executor(self):
        path = self.root / "executor-owner.json"
        if not path.exists():
            return
        owner = json.loads(path.read_text())
        if "started" not in owner:
            return
        process = subprocess.run(
            ["ps", "-p", str(owner["pid"]), "-o", "lstart=", "-o", "command="],
            capture_output=True,
            text=True,
        )
        if (
            process.returncode == 0
            and process.stdout.strip().startswith(owner["started"])
            and str(self.root) in process.stdout
        ):
            os.killpg(owner["pid"], signal.SIGKILL)
        path.unlink(missing_ok=True)

    def await_reply(self, process, deadline):
        while not select.select([process.stdout], [], [], 0.2)[0]:
            if process.poll() is not None:
                raise RuntimeError("owned execution process exited; see executor.log")
            if self.cancel_requested.is_set():
                self.stop_executor()
                raise InterruptedError("request cancelled")
            if time.monotonic() > deadline or self.shutdown_requested:
                self.stop_executor()
                raise TimeoutError("measurement exceeded its declared deadline or worker stopped")
        try:
            return receive(process.stdout)
        except EOFError as exc:
            code = process.wait(timeout=5)
            raise RuntimeError(
                f"numerical process exited with code {code}; see executor.log"
            ) from exc

    def numerical_process(self, source_id, store, deadline=None):
        if self.executor and self.executor.poll() is None and self.executor_source == source_id:
            return self.executor
        source = store.source(source_id)
        if self.executor and self.executor.poll() is None and self.executor_source:
            from .pairing import changed_paths, replaceable

            changed = changed_paths(store.source(self.executor_source), source)
            if changed and all(replaceable(path) for path in changed):
                assert self.executor_directory is not None
                entries = {f.path: f for f in source.files}
                for path in changed:
                    output = self.executor_directory / path
                    if path in entries:
                        atomic_write(output, store.blob(entries[path].blob))
                    else:
                        output.unlink(missing_ok=True)
                (self.executor_directory / ".ready").unlink(missing_ok=True)
                self.executor_source = source_id
                self.refreshed = True
                return self.executor
        self.stop_executor()
        with (self.root / "executor.log").open("ab") as log:
            preparation = self.own_process(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "roofline.environment",
                        "--root",
                        str(self.root),
                        "--source",
                        source_id,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log,
                    start_new_session=True,
                )
            )
            prepared = self.await_reply(preparation, deadline or time.monotonic() + 1800)
            preparation.wait(timeout=5)
            if "error" in prepared:
                raise RuntimeError(prepared["error"])
            destination, python, env = (
                Path(prepared["destination"]),
                prepared["python"],
                prepared["env"],
            )
            process = subprocess.Popen(
                [str(python), "-S", "-m", "roofline.executor", "--root", str(self.root)],
                cwd=destination,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                start_new_session=True,
            )
        self.own_process(process)
        self.executor_source = source_id
        self.executor_directory = destination
        self.refreshed = False
        return process

    def work(self, record):
        request = Request.model_validate(record["request"])
        attempt = Attempt.model_validate(record["attempt"])
        try:
            with Store(self.root) as store:
                if record.get("cancelled"):
                    raise InterruptedError("request cancelled")
                for name, blob in request.inputs.items():
                    if name == "component":
                        continue
                    if name != "prose.moby-dick":
                        raise ValueError(f"unsupported fixture input: {name}")
                    atomic_write(
                        self.root / "cache/magnitude/benchmarks/sources" / blob / "moby-dick.txt",
                        store.blob(blob),
                    )
                deadline = time.monotonic() + request.experiment.protocol.deadline_seconds
                numerical = request
                if attempt.role == "inputs":
                    numerical = request.model_copy(
                        update={
                            "operation": "prepare-inputs",
                            "experiment": request.experiment.model_copy(
                                update={
                                    "source": request.experiment.input_source,
                                }
                            ),
                        }
                    )
                process = self.numerical_process(numerical.experiment.source, store, deadline)
                if self.cancel_requested.is_set():
                    raise InterruptedError("request cancelled")
                # Delivery state belongs to the supervisor. Numerical owners
                # receive immutable experiment and target definitions only.
                definition = numerical.model_dump(exclude={"inputs", "cancelled", "attempts"})
                definition["experiment"] = numerical.experiment.model_dump(exclude_none=True)
                if "component" in request.inputs:
                    definition["inputs"] = {"component": request.inputs["component"]}
                target_fields = {"attempt_id", "target_name", "target"}
                definition["attempts"] = [
                    a.model_dump(include=target_fields) for a in request.attempts
                ]
                send(
                    process.stdin,
                    {
                        "request": definition,
                        "attempt": attempt.model_dump(include=target_fields),
                        "refresh": self.refreshed,
                    },
                )
                self.refreshed = False
                reply = self.await_reply(process, deadline)
                if "error" in reply:
                    if reply.get("code") == "NotImplementedError":
                        raise NotImplementedError(reply["error"])
                    raise RuntimeError(reply["error"])
                measurements = [Measurement.model_validate(m) for m in reply["result"]]
                measurements = [
                    m.model_copy(
                        update={
                            "artifacts": {**m.artifacts, **request.inputs},
                            "details": {
                                **m.details,
                                "model_definition": request.model.model_dump()
                                if request.model
                                else None,
                            },
                        }
                    )
                    for m in measurements
                ]
                for m in measurements:
                    store.put("measurement", m.measurement_id, m)
                record.update(
                    artifacts=reply.get("artifacts", []),
                    status="complete",
                    measurements=[m.model_dump() for m in measurements],
                )
        except Exception as exc:
            traceback.print_exc()
            self.stop_executor()
            with Store(self.root) as store:
                log = self.root / "executor.log"
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 65536))
                        artifact = store.put_blob(stream.read())
                    record["artifacts"] = [artifact]
                recovered = []
                for saved in store.records("measurement-checkpoint"):
                    if saved["attempt_id"] == attempt.attempt_id:
                        m = Measurement.model_validate(saved).model_copy(update={"error": str(exc)})
                        store.put("measurement", m.measurement_id, m)
                        recovered.append(m.model_dump())
                record["measurements"] = recovered
            record.update(
                status="cancelled"
                if record.get("cancelled")
                else "unavailable"
                if isinstance(exc, NotImplementedError)
                else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self.lock, Store(self.root) as store:
                latest = store.get("attempt", attempt.attempt_id)
                if latest.get("cancelled"):
                    record["status"] = "cancelled"
                store.put("attempt", attempt.attempt_id, record, mutable=True)
                self.active = None
                self.last_used = time.monotonic()

    def coordinate(self, request_id, attempt_id):
        disconnected_since = None
        try:
            while not self.shutdown_requested:
                with Store(self.root) as store:
                    request = store.request(request_id)
                    attempt = next(a for a in request.attempts if a.attempt_id == attempt_id)
                    if attempt.status in TERMINAL:
                        return
                try:
                    with (
                        WorkerClient(attempt.target, self.project) as client,
                        Store(self.root) as store,
                    ):
                        # Reconcile before transferring: an accepted worker may be
                        # sampling, and source transfer must not block its collection.
                        try:
                            client.call({"op": "status", "attempt_id": attempt_id})
                            attempt = attempt.model_copy(update={"dispatched": True})
                        except WorkerError as exc:
                            if exc.code != "KeyError":
                                raise
                            if request.cancelled:
                                self.update_attempt(
                                    request_id, attempt.model_copy(update={"status": "cancelled"})
                                )
                                return
                            transfer_source(client, store, store.source(request.experiment.source))
                            if request.experiment.against_source:
                                transfer_source(
                                    client, store, store.source(request.experiment.against_source)
                                )
                            if request.experiment.input_source:
                                transfer_source(
                                    client, store, store.source(request.experiment.input_source)
                                )
                            for blob in request.inputs.values():
                                content = store.blob(blob)
                                for offset in range(0, len(content), 1 << 20):
                                    client.call(
                                        {
                                            "op": "blob",
                                            "id": blob,
                                            "offset": offset,
                                            "content": base64.b64encode(
                                                content[offset : offset + (1 << 20)]
                                            ).decode(),
                                            "final": offset + (1 << 20) >= len(content),
                                        }
                                    )
                            attempt = attempt.model_copy(update={"dispatched": True})
                            self.update_attempt(request_id, attempt)
                            client.call(
                                {
                                    "op": "accept",
                                    "request": request.model_dump(),
                                    "attempt": attempt.model_dump(),
                                }
                            )
                        disconnected_since = None
                        while not self.shutdown_requested:
                            with Store(self.root) as fresh:
                                if fresh.request(request_id).cancelled:
                                    client.call({"op": "cancel", "attempt_id": attempt_id})
                            status = client.call({"op": "status", "attempt_id": attempt_id})
                            if status["status"] in TERMINAL:
                                for blob in status.get("artifacts", []):
                                    collect_blob(client, store, blob)
                                measurements = [
                                    Measurement.model_validate(m) for m in status["measurements"]
                                ]
                                for m in measurements:
                                    for blob in m.artifacts.values():
                                        collect_blob(client, store, blob)
                                    store.put("measurement", m.measurement_id, m)
                                updated = attempt.model_copy(
                                    update={
                                        "status": status["status"],
                                        "error": status["error"],
                                        "artifact_ids": tuple(status.get("artifacts", [])),
                                        "measurement_ids": tuple(
                                            m.measurement_id for m in measurements
                                        ),
                                    }
                                )
                                self.update_attempt(request_id, updated)
                                return
                            attempt = attempt.model_copy(
                                update={"status": status["status"], "error": None}
                            )
                            self.update_attempt(request_id, attempt)
                            time.sleep(0.5)
                except WorkerError as exc:
                    if str(exc) not in {"worker busy", "worker paused"}:
                        raise
                    self.update_attempt(request_id, attempt.model_copy(update={"error": str(exc)}))
                    time.sleep(3)
                except (OSError, EOFError) as exc:
                    # Reconnect with the same attempt ID. Never turn an uncertain
                    # submission into a second execution under a fresh ID.
                    if disconnected_since is None:
                        disconnected_since = time.monotonic()
                    self.update_attempt(request_id, attempt.model_copy(update={"error": str(exc)}))
                    if (
                        not attempt.dispatched
                        and time.monotonic() - disconnected_since
                        > request.experiment.protocol.deadline_seconds
                    ):
                        self.update_attempt(
                            request_id,
                            attempt.model_copy(update={"status": "unavailable", "error": str(exc)}),
                        )
                        return
                    time.sleep(3)
        except Exception as exc:
            traceback.print_exc()
            with Store(self.root) as store:
                request = store.request(request_id)
                attempt = next(a for a in request.attempts if a.attempt_id == attempt_id)
            self.update_attempt(
                request_id, attempt.model_copy(update={"status": "failed", "error": str(exc)})
            )
        finally:
            with self.lock:
                self.running.discard(attempt_id)

    def update_attempt(self, request_id, attempt):
        with self.lock, Store(self.root) as store:
            request = store.request(request_id)
            inputs = dict(request.inputs)
            if attempt.role == "inputs" and attempt.status == "complete":
                if len(attempt.artifact_ids) != 1:
                    raise ValueError("input preparation must return exactly one verified boundary")
                inputs["component"] = attempt.artifact_ids[0]
            store.put(
                "request",
                request_id,
                request.model_copy(
                    update={
                        "inputs": inputs,
                        "attempts": tuple(
                            attempt if a.attempt_id == attempt.attempt_id else a
                            for a in request.attempts
                        ),
                    }
                ),
                mutable=True,
            )

    def schedule(self):
        # A previous supervisor cannot leave an apparently live execution after restart.
        if self.kind == "worker":
            self.reap_previous_executor()
            with Store(self.root) as store:
                for record in store.records("attempt"):
                    if record["status"] == "running":
                        record.update(status="failed", error="worker restarted during execution")
                        store.put("attempt", record["attempt"]["attempt_id"], record, mutable=True)
        while not self.shutdown_requested:
            try:
                with self.lock, Store(self.root) as store:
                    if self.kind == "coordinator":
                        for record in store.records("request"):
                            if all(a["status"] in TERMINAL for a in record["attempts"]):
                                continue
                            request = Request.model_validate(record)
                            producer = next(
                                (a for a in request.attempts if a.role == "inputs"), None
                            )
                            for attempt in request.attempts:
                                if producer and attempt.role == "measure":
                                    if producer.status not in TERMINAL:
                                        continue
                                    if (
                                        producer.status != "complete"
                                        and attempt.status not in TERMINAL
                                    ):
                                        self.update_attempt(
                                            request.request_id,
                                            attempt.model_copy(
                                                update={
                                                    "status": "cancelled"
                                                    if request.cancelled
                                                    else "unavailable",
                                                    "error": (
                                                        "shared input preparation did not complete"
                                                    ),
                                                }
                                            ),
                                        )
                                        continue
                                if (
                                    attempt.status not in TERMINAL
                                    and attempt.attempt_id not in self.running
                                ):
                                    self.running.add(attempt.attempt_id)
                                    self.pool.submit(
                                        self.coordinate, request.request_id, attempt.attempt_id
                                    )
                    elif not self.active:
                        pending = [r for r in store.records("attempt") if r["status"] == "accepted"]
                        if pending:
                            record = pending[0]
                            record["status"] = "running"
                            self.cancel_requested.clear()
                            self.active = record["attempt"]["attempt_id"]
                            store.put("attempt", self.active, record, mutable=True)
                            self.pool.submit(self.work, record)
                        elif self.executor and time.monotonic() - self.last_used > 300:
                            self.stop_executor()
            except Exception:
                traceback.print_exc()
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--kind", choices=("worker", "coordinator"), default="worker")
    parser.add_argument("--bridge", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    project = args.project.expanduser().resolve() if args.project else None
    if args.bridge:
        ensure_service(root, project, "worker")
        while True:
            try:
                message = receive(sys.stdin.buffer)
            except EOFError:
                return
            try:
                send(
                    sys.stdout.buffer,
                    {
                        "result": local_call(
                            root, message, timeout=3600 if message["op"] == "discover" else 60
                        )
                    },
                )
            except Exception as exc:
                send(sys.stdout.buffer, failure(exc))
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    with (root / "service.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        path = socket_path(root)
        path.unlink(missing_ok=True)
        service = Service(root, project, args.kind)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    send(self.wfile, {"result": service.dispatch(receive(self.rfile))})
                    if service.shutdown_requested:
                        threading.Thread(target=server.shutdown, daemon=True).start()
                except Exception as exc:
                    send(self.wfile, failure(exc))

        with socketserver.ThreadingUnixStreamServer(str(path), Handler) as server:
            os.chmod(path, 0o600)
            threading.Thread(target=service.schedule, daemon=True).start()

            def shutdown(signum, frame):
                service.shutdown_requested = True
                threading.Thread(target=server.shutdown, daemon=True).start()

            signal.signal(signal.SIGTERM, shutdown)
            signal.signal(signal.SIGINT, shutdown)
            try:
                server.serve_forever()
            finally:
                service.stop_executor()
                service.pool.shutdown(wait=False)
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
