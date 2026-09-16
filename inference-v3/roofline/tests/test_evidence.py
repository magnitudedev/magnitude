import io
import sys
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

from roofline.bundles import export_bundle, import_bundle
from roofline.contracts import Measurement, Models, Source, SourceFile
from roofline.query import Queries, compare
from roofline.store import Store
from roofline.transport import receive, send


def measurement(index=0, **changes):
    values = dict(
        measurement_id=str(index),
        request_id="request",
        attempt_id="attempt",
        created=(datetime(2026, 9, 15, tzinfo=UTC) + timedelta(seconds=index)).isoformat(),
        model="qwen:gguf:q4",
        artifact="a" * 64,
        source_id="b" * 64,
        target="local",
        engine="magnitude",
        workload={"context": 16},
        scope="decode",
        hardware={"device": "physical-device-a"},
        protocol={"samples": 3},
        status="complete",
        correctness="passed",
        samples_seconds=(0.1, 0.11, 0.12),
    )
    values.update(changes)
    return Measurement(**values)


def test_query_does_not_import_numerical_runtime(tmp_path):
    before = set(sys.modules)
    with Store(tmp_path / "absent", readonly=True) as store:
        assert store.records("measurement") == []
    assert not (tmp_path / "absent").exists()
    assert not ({"ops", "torch", "tilelang", "engine"} & (set(sys.modules) - before))


def test_identity_and_finite_observations():
    with pytest.raises(ValueError):
        Models.model_validate(
            {
                "models": {
                    "qwen": {
                        "sha256": "a" * 64,
                        "locations": {"local": "/models/model.gguf"},
                    }
                }
            }
        )
    with pytest.raises(ValueError):
        measurement(samples_seconds=(float("nan"),))
    with pytest.raises(ValueError):
        SourceFile(path="../escape", blob="a" * 64)


def test_history_preserves_failures_and_groups_hardware(tmp_path):
    with Store(tmp_path) as store:
        for m in (
            measurement(),
            measurement(1, correctness="failed", status="failed"),
            measurement(2, hardware={"device": "physical-device-b"}),
        ):
            store.put("measurement", m.measurement_id, m)
        report = Queries(store).model("qwen:gguf:q4")
        assert report["condition_count"] == 2
        group = next(g for g in report["groups"] if g["measurement_count"] == 2)
        assert group["latest"]["correctness"] == "failed"
        assert group["best_correct"]["measurement_id"] == "0"
        assert not group["change"]["qualified"]
        with pytest.raises(ValueError, match="collision"):
            store.put("measurement", "0", measurement(samples_seconds=(0.2, 0.2, 0.2)))


def test_pagination_uses_fixed_snapshot_and_complete_counts(tmp_path):
    with Store(tmp_path) as store:
        for i in range(30):
            m = measurement(i, workload={"context": i})
            store.put("measurement", m.measurement_id, m)
        queries = Queries(store)
        first = queries.model("qwen:gguf:q4")
        assert first["measurement_count"] == 30
        assert len(first["groups"]) == 24
        store.put("measurement", "40", measurement(40, workload={"context": 40}))
        second = queries.model("qwen:gguf:q4", cursor=first["cursor"])
        assert second["measurement_count"] == 30
        assert len(second["groups"]) == 6
        with pytest.raises(ValueError):
            queries.model("another", cursor=first["cursor"])


def test_comparison_requires_real_pair_identity():
    a, b = measurement(), measurement(1, samples_seconds=(0.07, 0.08, 0.09))
    assert compare(a, b)["evidence"] == "historical"
    assert compare(a, b)["latency_change_percent"] < 0
    assert not compare(a, b.model_copy(update={"artifact": "c" * 64}))["compatible"]
    assert (
        compare(
            a.model_copy(update={"pair_ids": ("a", "b", "c")}),
            b.model_copy(update={"pair_ids": ("a", "b", "c")}),
        )["evidence"]
        == "paired"
    )


def test_portable_bundle_integrity_and_idempotency(tmp_path):
    with Store(tmp_path / "original") as store:
        blob = store.put_blob(b"authored source")
        source = Source(files=(SourceFile(path="src/example.py", blob=blob),))
        store.put("source", source.source_id, source)
        m = measurement(source_id=source.source_id, artifacts={"source": blob})
        store.put("measurement", m.measurement_id, m)
        bundle = tmp_path / "evidence.zip"
        export_bundle(store, [m.measurement_id], bundle)
    with Store(tmp_path / "imported") as imported:
        import_bundle(imported, bundle)
        import_bundle(imported, bundle)
        assert len(imported.records("measurement")) == 1
        assert imported.blob(blob) == b"authored source"
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bundle) as old, zipfile.ZipFile(bad, "w") as new:
            for name in old.namelist():
                new.writestr(name, b"tampered" if name.startswith("blobs/") else old.read(name))
        with pytest.raises(ValueError, match="checksum"):
            import_bundle(imported, bad)


def test_framing_and_bounded_artifact_pages(tmp_path):
    stream = io.BytesIO()
    send(stream, {"value": "unicode ✓"})
    stream.seek(0)
    assert receive(stream) == {"value": "unicode ✓"}
    with Store(tmp_path) as store:
        content = ("unicode ✓" * 5000).encode()
        blob = store.put_blob(content)
        query = Queries(store)
        page, recovered = query.artifact(blob), bytearray()
        while True:
            import base64

            recovered.extend(
                page["content"].encode()
                if page["encoding"] == "utf-8"
                else base64.b64decode(page["content"])
            )
            if not page["cursor"]:
                break
            page = query.artifact(blob, cursor=page["cursor"])
        assert bytes(recovered) == content


def test_component_history_requires_equal_logical_inputs(tmp_path):
    first = measurement(
        scope="decode/block[0]/ffn",
        details={
            "comparison": {
                "semantics": "same-formula",
                "fixture": "producer-a",
                "geometry": "same-shapes",
            }
        },
    )
    second = measurement(
        1,
        scope=first.scope,
        details={
            "comparison": {
                "semantics": "same-formula",
                "fixture": "producer-b",
                "geometry": "same-shapes",
            }
        },
    )
    result = compare(first, second)
    assert not result["compatible"]
    assert result["delta_seconds"] is None
    assert "component_contract" in result["differences"]
    with Store(tmp_path) as store:
        for m in (first, second):
            store.put("measurement", m.measurement_id, m)
        assert Queries(store).model(first.model)["condition_count"] == 2


def test_shared_input_export_includes_the_producer_source(tmp_path):
    with Store(tmp_path / "store") as store:
        producer = Source(files=())
        file = SourceFile(path="src/consumer.py", blob=store.put_blob(b"consumer source"))
        consumer = Source(files=(file,))
        for source in (producer, consumer):
            store.put("source", source.source_id, source)
        m = measurement(
            source_id=consumer.source_id,
            workload={
                "input_policy": {
                    "kind": "frozen-shared-boundary",
                    "source_id": producer.source_id,
                }
            },
        )
        store.put("measurement", m.measurement_id, m)
        path = tmp_path / "shared.zip"
        export_bundle(store, [m.measurement_id], path)
    with Store(tmp_path / "imported") as store:
        import_bundle(store, path)
        assert store.source(producer.source_id) == producer
        assert store.source(consumer.source_id) == consumer
