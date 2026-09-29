import json
from types import SimpleNamespace

from roofline.bundles import export_bundle, import_bundle, import_session
from roofline.contracts import Model, digest
from roofline.query import Queries
from roofline.store import Store


def test_session_import_preserves_timers_chronology_and_unknown_source(tmp_path):
    directory = tmp_path / "session"
    directory.mkdir()
    reference = "/old/model.gguf"
    target = "llama.cpp-" + digest(reference)[:10]
    records = {
        "run.json": {
            "format": 1,
            "started_at": "2020-01-01T00:00:00+00:00",
            "selection": {"targets": [{"engine": "llama.cpp", "reference": reference}]},
            "host": {"hardware": {"chip": "old machine"}},
        },
        target + "-artifact.json": {
            "kind": "gguf",
            "reference": reference,
            "files": [{"sha256": "a" * 64}],
        },
        "requests.jsonl": {"id": "request", "context": 128},
        "results.jsonl": {
            "phase": "measured",
            "block": 0,
            "target": target,
            "timing_basis": "native-service",
            "observation": {
                "request_id": "request",
                "outcome": "invalid",
                "ttft_ms": 12,
                "completed_ms": 50,
                "terminal": {
                    "timings": {
                        "prompt_ms": 10,
                        "prompt_n": 128,
                        "predicted_ms": 30,
                        "predicted_n": 2,
                    }
                },
            },
        },
    }
    for name, value in records.items():
        (directory / name).write_text(json.dumps(value))
    config = SimpleNamespace(models={"test:gguf:q4": Model(sha256="a" * 64, locations={})})
    with Store(tmp_path / "store") as store:
        report = import_session(store, config, directory)
        assert report["imported"] == 4 and not report["unresolved"]
        assert import_session(store, config, directory) == report
        assert len(store.records("measurement")) == 4
        m = store.measurement(report["measurement_ids"][-1])
        assert m.samples_seconds == (0.03,)
        assert m.source_id is None and m.correctness == "unchecked"
        assert m.created.startswith("2020")
        assert m.protocol["boundary"] == "native-service:predicted_ms"
        assert m.workload["count"] == 2
        assert m.details["session_outcome"] == "invalid"
        assert all(g["best_correct"] is None for g in Queries(store).model(m.model)["groups"])
        archive = tmp_path / "session.zip"
        export_bundle(store, report["measurement_ids"], archive)
    with Store(tmp_path / "copy") as store:
        import_bundle(store, archive)
        assert len(store.records("measurement")) == 4
    with Store(tmp_path / "unmatched") as store:
        report = import_session(store, SimpleNamespace(models={}), directory)
        assert report["imported"] == 0 and report["unresolved"]
        assert Queries(store).artifact(report["artifact_id"])["content"]
