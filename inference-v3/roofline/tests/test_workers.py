import base64
import hashlib

import pytest

from roofline.contracts import (
    Attempt,
    Connection,
    Device,
    Experiment,
    Request,
    Source,
    Target,
)
from roofline.service import Service
from roofline.sources import capture, materialize
from roofline.store import Store


def request():
    target = Target(connection=Connection(kind="local"), device=Device(backend="cpu"))
    attempt = Attempt(attempt_id="attempt", target_name="local", target=target)
    return Request(
        request_id="request",
        created="2026-09-15T00:00:00+00:00",
        model=None,
        operation="characterize",
        experiment=Experiment(model="resources", source=Source(files=()).source_id),
        attempts=(attempt,),
    ), attempt


def test_worker_acceptance_is_durable_and_idempotent(tmp_path):
    service = Service(tmp_path, tmp_path / "project", "worker")
    source = Source(files=())
    with Store(tmp_path) as store:
        store.put("source", source.source_id, source)
    req, attempt = request()
    message = {"op": "accept", "request": req.model_dump(), "attempt": attempt.model_dump()}
    assert service.dispatch(message) == "accepted"
    changed = req.model_copy(
        update={"attempts": (attempt.model_copy(update={"status": "running"}),)}
    )
    assert service.dispatch({**message, "request": changed.model_dump()}) == "accepted"
    with Store(tmp_path) as store:
        assert len(store.records("attempt")) == 1
    service.dispatch({"op": "cancel", "attempt_id": "attempt"})
    assert service.dispatch({"op": "status", "attempt_id": "attempt"})["cancelled"]
    service.dispatch({"op": "pause"})
    assert Service(tmp_path, tmp_path / "project", "worker").paused


def test_content_delivery_detects_corruption_and_busy_workers(tmp_path):
    service = Service(tmp_path, tmp_path, "worker")
    data = b"a source file"
    blob = hashlib.sha256(data).hexdigest()
    frame = {
        "op": "blob",
        "id": blob,
        "offset": 0,
        "content": base64.b64encode(data).decode(),
        "final": True,
    }
    service.active = "another measurement"
    with pytest.raises(RuntimeError, match="busy"):
        service.dispatch(frame)
    service.active = None
    assert service.dispatch(frame)
    assert service.dispatch({"op": "missing", "blobs": [blob]}) == []
    with pytest.raises(ValueError, match="checksum"):
        service.dispatch({**frame, "id": "0" * 64})


def test_snapshot_reconstructs_deletions_and_rejects_external_links(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    path = project / "src/main.py"
    path.write_text("version = 1\n")
    with Store(tmp_path / "store") as store:
        first = capture(project, store)
        path.write_text("version = 2\n")
        second = capture(project, store)
        assert first.source_id != second.source_id
        destination = tmp_path / "execution"
        materialize(first, store, destination)
        assert (destination / "src/main.py").read_text() == "version = 1\n"
        path.unlink()
        third = capture(project, store)
        materialize(third, store, destination)
        assert not (destination / "src/main.py").exists()
        external = tmp_path / "outside.py"
        external.write_text("outside")
        path.symlink_to(external)
        with pytest.raises(ValueError, match="leaves inference-v3"):
            capture(project, store)


def test_missing_environment_publishes_failure_without_numerical_evidence(tmp_path):
    service = Service(tmp_path, tmp_path / "missing-project", "worker")
    source = Source(files=())
    with Store(tmp_path) as store:
        store.put("source", source.source_id, source)
    req, attempt = request()
    service.dispatch({"op": "accept", "request": req.model_dump(), "attempt": attempt.model_dump()})
    service.work(service.dispatch({"op": "status", "attempt_id": attempt.attempt_id}))
    result = service.dispatch({"op": "status", "attempt_id": attempt.attempt_id})
    assert result["status"] == "failed"
    assert result["measurements"] == []
    assert "missing its dependency lock" in result["error"]


def test_execution_environment_never_uses_checkout_dependencies(tmp_path, monkeypatch):
    from roofline.environment import prepare_environment

    worker = tmp_path / "worker"
    worker.mkdir()
    project = tmp_path / "unrelated-checkout"
    (project / ".venv/bin").mkdir(parents=True)
    (project / ".venv/bin/python").write_text("unrelated interpreter")
    (project / "src").mkdir()
    (project / "src/main.py").write_text("accepted source")
    (project / "uv.lock").write_text("accepted lock")
    (project / "hatch_build.py").write_text("build hook")
    (project / "native/templates").mkdir(parents=True)
    (project / "native/templates/library.cpp").write_text("native source")
    (project / "src/templates/_native").mkdir(parents=True)
    (project / "src/templates/_native/local-build.json").write_text("unrelated build")
    monkeypatch.setenv("VIRTUAL_ENV", str(project / ".venv"))
    monkeypatch.setenv("PYTHONPATH", str(project / "src"))
    monkeypatch.setenv("TVM_LIBRARY_PATH", str(project / "tilelang/build/lib"))
    calls = []

    def build(command, *, cwd, env, **kwargs):
        calls.append((command, cwd, env))
        assert command[0] == str(worker / "bin/uv")
        assert command[command.index("--group") + 1] == "performance"
        assert "VIRTUAL_ENV" not in env and "PYTHONPATH" not in env
        assert "TVM_LIBRARY_PATH" not in env
        assert env["UV_CACHE_DIR"].startswith(str(worker))
        python = cwd / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        (cwd / "tilelang/build").mkdir(parents=True)
        library = cwd / "src/templates/_native/library.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"worker-built library")

    monkeypatch.setattr("roofline.environment.subprocess.run", build)
    with Store(worker) as store:
        source = capture(project, store)
        paths = {file.path for file in source.files}
        assert {"hatch_build.py", "native/templates/library.cpp"} <= paths
        assert not any("/_native/" in path for path in paths)
        destination, python, env = prepare_environment(worker, source.source_id, store)
        assert destination.is_relative_to(worker)
        assert python.is_relative_to(worker)
        assert str(project) not in env["PYTHONPATH"]
        assert env["TVM_LIBRARY_PATH"].startswith(str(worker))
        prepare_environment(worker, source.source_id, store)
        assert len(calls) == 1
        (project / "src/main.py").write_text("changed engine code")
        changed = capture(project, store)
        new_destination, reused_python, new_env = prepare_environment(
            worker, changed.source_id, store
        )
        assert new_destination != destination
        assert reused_python == python
        assert (new_destination / "tilelang/build").is_symlink()
        assert (new_destination / "tilelang/build").resolve() == destination / "tilelang/build"
        assert (new_destination / "src/templates/_native/library.so").read_bytes() == (
            b"worker-built library"
        )
        assert new_env["PYTHONPATH"].startswith(str(new_destination))
        assert len(calls) == 1
        (project / "uv.lock").write_text("changed dependency lock")
        changed_lock = capture(project, store)
        _, rebuilt_python, _ = prepare_environment(worker, changed_lock.source_id, store)
        assert rebuilt_python != python
        assert len(calls) == 2
        (project / "native/templates/library.cpp").write_text("changed native source")
        changed_native = capture(project, store)
        _, native_python, _ = prepare_environment(worker, changed_native.source_id, store)
        assert native_python != rebuilt_python
        assert len(calls) == 3
    assert (project / ".venv/bin/python").read_text() == "unrelated interpreter"


def test_cold_preflight_reports_preparation_without_build_or_source_transfer(tmp_path):
    from roofline.contracts import Model

    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"metadata will be checked during discovery")
    model = Model(sha256="a" * 64, locations={"local": str(artifact)})
    target = Target(connection=Connection(kind="local"), device=Device(backend="cpu"))
    service = Service(tmp_path / "worker", None, "worker")
    message = {
        "op": "preflight",
        "dependency_id": "b" * 64,
        "payload": {
            "model": model.model_dump(),
            "target": target.model_dump(),
            "experiment": Experiment(model="test:gguf:q4").model_dump(),
        },
    }
    result = service.dispatch(message)
    assert result["status"] == "preparation-required"
    assert not result["environment_ready"] and not result["artifact_verified"]
    assert not (service.root / "executors").exists()
    artifact.unlink()
    result = service.dispatch(message)
    assert result["status"] == "unavailable"
    assert "missing" in result["unresolved"][0]


def test_discovery_checks_every_target_independently(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from roofline.contracts import Model
    from roofline.discovery import discover

    names = ("first", "second")
    config = SimpleNamespace(
        root=tmp_path,
        workspace=tmp_path / "evidence",
        models={"test:gguf:q4": Model(sha256="a" * 64, locations={n: f"/{n}.gguf" for n in names})},
        targets={
            n: Target(connection=Connection(kind="ssh", host=n), device=Device(backend="cuda"))
            for n in names
        },
    )
    seen = set()

    class Client:
        def __init__(self, target, project):
            self.name = target.connection.host

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def call(self, message):
            assert message["op"] == "preflight"
            assert message["payload"]["experiment"]["targets"] == (self.name,)
            seen.add(self.name)
            return {
                "status": "preparation-required",
                "environment_ready": False,
                "scopes": [],
                "unresolved": [],
            }

    monkeypatch.setattr("roofline.transport.WorkerClient", Client)
    result = discover(config, Experiment(model="test:gguf:q4", targets=names))
    assert seen == set(names)
    assert all(r["status"] == "preparation-required" for r in result["targets"].values())


def test_worker_update_refuses_accepted_or_active_work(tmp_path):
    service = Service(tmp_path, None, "worker")
    service.active = "measurement"
    with pytest.raises(RuntimeError, match="busy"):
        service.dispatch({"op": "shutdown"})
    service.active = None
    source = Source(files=())
    with Store(tmp_path) as store:
        store.put("source", source.source_id, source)
    req, attempt = request()
    service.dispatch({"op": "accept", "request": req.model_dump(), "attempt": attempt.model_dump()})
    with pytest.raises(RuntimeError, match="busy"):
        service.dispatch({"op": "shutdown"})
    assert not service.shutdown_requested


def test_small_source_files_transfer_in_verified_batches(tmp_path):
    from roofline.service import transfer_source

    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    for index in range(100):
        (project / f"src/file{index}.py").write_text(f"value={index}")
    service = Service(tmp_path / "worker", None, "worker")
    operations = []

    class Client:
        def call(self, message):
            operations.append(message["op"])
            return service.dispatch(message)

    with Store(tmp_path / "coordinator") as store:
        source = capture(project, store)
        transfer_source(Client(), store, source)
        assert operations == ["missing", "blobs", "source"]
        with Store(tmp_path / "worker") as remote:
            for file in source.files:
                assert remote.blob(file.blob) == store.blob(file.blob)
        operations.clear()
        transfer_source(Client(), store, source)
        assert operations == ["missing", "source"]
    with pytest.raises(ValueError, match="checksum"):
        service.dispatch({"op": "blobs", "contents": {"0" * 64: "dGFtcGVyZWQ="}})


def test_snapshot_preserves_compiler_build_inputs_in_testing_directories(tmp_path):
    project = tmp_path / "project"
    source = project / "tilelang/3rdparty/tvm/3rdparty/tvm-ffi/src/ffi/testing/testing.cc"
    source.parent.mkdir(parents=True)
    source.write_text("// required by compiler CMake build")
    generated = project / "tilelang/build/generated.cc"
    generated.parent.mkdir()
    generated.write_text("// generated")
    with Store(tmp_path / "store") as store:
        snapshot = capture(project, store)
        paths = {entry.path for entry in snapshot.files}
        assert source.relative_to(project).as_posix() in paths
        assert generated.relative_to(project).as_posix() not in paths


def test_bridge_preserves_worker_failure_classification():
    from roofline.transport import WorkerError, failure

    reply = failure(KeyError("missing attempt"))
    bridged = failure(WorkerError(reply["error"], reply["code"]))
    assert bridged == reply


def test_reconciliation_collects_without_resubmission_or_source_transfer(tmp_path, monkeypatch):
    service = Service(tmp_path, None, "coordinator")
    req, attempt = request()
    with Store(tmp_path) as store:
        store.put("request", req.request_id, req)
    operations = []

    class Client:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def call(self, message):
            operations.append(message["op"])
            assert message["op"] == "status"
            return {"status": "complete", "artifacts": [], "measurements": [], "error": None}

    monkeypatch.setattr("roofline.service.WorkerClient", Client)
    service.coordinate(req.request_id, attempt.attempt_id)
    with Store(tmp_path) as store:
        assert store.request(req.request_id).attempts[0].status == "complete"
    assert operations == ["status", "status"]


def test_permanent_worker_rejection_is_not_retried(tmp_path, monkeypatch):
    from roofline.transport import WorkerError

    service = Service(tmp_path, None, "coordinator")
    req, attempt = request()
    with Store(tmp_path) as store:
        store.put("request", req.request_id, req)

    def rejected(*args):
        raise WorkerError("unsupported worker protocol", "ValueError")

    monkeypatch.setattr("roofline.service.WorkerClient", rejected)
    service.coordinate(req.request_id, attempt.attempt_id)
    with Store(tmp_path) as store:
        result = store.request(req.request_id).attempts[0]
        assert result.status == "failed" and "protocol" in result.error


def test_source_restoration_preserves_built_dependencies(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    source_file = project / "src/main.py"
    source_file.write_text("first")
    destination = tmp_path / "execution"
    with Store(tmp_path / "store") as store:
        first = capture(project, store)
        materialize(first, store, destination)
        library = destination / ".venv/lib/native.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"compiled dependency")
        source_file.unlink()
        second = capture(project, store)
        materialize(second, store, destination)
        assert not (destination / "src/main.py").exists()
        assert library.read_bytes() == b"compiled dependency"


def test_owned_preparation_deadline_terminates_process_group(tmp_path):
    import subprocess
    import sys
    import time

    service = Service(tmp_path, None, "worker")
    process = service.own_process(
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
    )
    with pytest.raises(TimeoutError, match="deadline"):
        service.await_reply(process, time.monotonic())
    assert process.poll() is not None


def test_executor_failure_recovers_completed_samples(tmp_path, monkeypatch):
    from test_evidence import measurement

    service = Service(tmp_path, None, "worker")
    req, attempt = request()
    with Store(tmp_path) as store:
        store.put("source", Source(files=()).source_id, Source(files=()))
    service.dispatch({"op": "accept", "request": req.model_dump(), "attempt": attempt.model_dump()})

    def crash(source, store, deadline):
        checkpoint = measurement(status="incomplete", correctness="unchecked", source_id=source)
        store.put("measurement-checkpoint", checkpoint.measurement_id, checkpoint)
        raise RuntimeError("executor crashed during diagnostics")

    monkeypatch.setattr(service, "numerical_process", crash)
    service.work(service.dispatch({"op": "status", "attempt_id": attempt.attempt_id}))
    result = service.dispatch({"op": "status", "attempt_id": attempt.attempt_id})
    assert result["status"] == "failed"
    assert len(result["measurements"]) == 1
    recovered = result["measurements"][0]
    assert recovered["samples_seconds"] == [0.1, 0.11, 0.12]
    assert recovered["correctness"] == "unchecked"
    assert "diagnostics" in recovered["error"]


def test_discovery_publishes_complete_structure_before_paginating(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from roofline.contracts import Model
    from roofline.discovery import discover
    from roofline.model_view import build_tree

    config = SimpleNamespace(
        root=tmp_path,
        workspace=tmp_path / "evidence",
        models={"test:gguf:q4": Model(sha256="a" * 64, locations={"local": "/model.gguf"})},
        targets={
            "local": Target(connection=Connection(kind="local"), device=Device(backend="metal"))
        },
    )
    scopes = [
        {"selector": "decode/model[0]", "parent": "decode", "contract": "model"},
        *[
            {
                "selector": f"decode/model[0]/block[{i}]",
                "parent": "decode/model[0]",
                "contract": "block",
            }
            for i in range(40)
        ],
    ]

    class Client:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def call(self, message):
            if message["op"] == "preflight":
                return {"status": "ready", "environment_ready": True, "unresolved": []}
            assert message["op"] == "discover"
            from analytical import graph

            return {
                "scopes": scopes,
                "graph": "recorded-graph",
                "performance_manifest": graph(40).model_dump(mode="json"),
            }

    monkeypatch.setattr("roofline.transport.WorkerClient", Client)
    monkeypatch.setattr("roofline.service.transfer_source", lambda *_: None)
    result = discover(config, Experiment(model="test:gguf:q4", targets=("local",)))
    assert result["targets"]["local"]["scope_count"] == 41
    assert len(result["targets"]["local"]["scopes"]) == 24
    with Store(config.workspace, readonly=True) as store:
        tree = build_tree(store, "test:gguf:q4")
        assert len(tree.nodes) == 41
        assert all(not node.relation["points"] for node in tree.nodes.values())


def test_engine_environment_cannot_inherit_control_environment(tmp_path, monkeypatch):
    from roofline.environment import worker_environment

    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/another/environment")
    monkeypatch.setenv("PYTHONPATH", "/another/checkout")
    environment = worker_environment(tmp_path)
    assert "UV_PROJECT_ENVIRONMENT" not in environment
    assert "PYTHONPATH" not in environment
