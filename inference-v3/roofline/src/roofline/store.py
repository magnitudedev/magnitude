"""One local query authority with immutable records and content-addressed payloads."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path

from .contracts import Measurement, Request, Source, encoded


def atomic_write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path, *, readonly=False):
        self.root = Path(root)
        self.readonly = readonly
        path = self.root / "roofline.sqlite"
        if readonly and not path.exists():
            self.db = None
            return
        if not readonly:
            self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            f"file:{path}?mode=ro" if readonly else str(path), uri=readonly, timeout=30
        )
        self.db.row_factory = sqlite3.Row
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL, id TEXT NOT NULL, content BLOB NOT NULL,
                    UNIQUE(kind,id));
                CREATE TABLE IF NOT EXISTS measurements (
                    id TEXT PRIMARY KEY, model TEXT NOT NULL, target TEXT NOT NULL,
                    created TEXT NOT NULL, comparison_key TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS measurement_model ON measurements(model,created);
            """)

    def close(self):
        if self.db is not None:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get(self, kind, identity):
        row = (
            self.db.execute(
                "SELECT content FROM records WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
            if self.db
            else None
        )
        if row is None:
            raise KeyError(f"unknown {kind}: {identity}")
        import json

        return json.loads(row[0])

    def records(self, kind, *, cutoff=None):
        import json

        if self.db is None:
            return []
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT content FROM records WHERE kind=? "
                "AND (? IS NULL OR sequence<=?) ORDER BY sequence",
                (kind, cutoff, cutoff),
            )
        ]

    def model_names(self):
        """List models without decoding every measurement and its formula tree."""
        if self.db is None:
            return set()
        return {
            row[0]
            for row in self.db.execute(
                "SELECT DISTINCT model FROM measurements UNION "
                "SELECT json_extract(CAST(content AS TEXT), '$.model') FROM records "
                "WHERE kind='model-structure'"
            )
            if row[0] is not None
        }

    def put(self, kind, identity, record, *, mutable=False):
        self.put_many(((kind, identity, record),), mutable=mutable)

    def put_many(self, records, *, mutable=False):
        """Publish an entire evidence closure and its derived models atomically."""
        if self.readonly or self.db is None:
            raise RuntimeError("read-only evidence store")
        with self.db:
            changed = {}
            for kind, key, record in records:
                content = encoded(record)
                previous = self.db.execute(
                    "SELECT content FROM records WHERE kind=? AND id=?", (kind, key)
                ).fetchone()
                if previous:
                    if bytes(previous[0]) == content:
                        continue
                    if not mutable:
                        raise ValueError(f"immutable {kind} identity collision: {key}")
                self.db.execute(
                    "INSERT INTO records(kind,id,content) VALUES(?,?,?) "
                    "ON CONFLICT(kind,id) DO UPDATE SET content=excluded.content",
                    (kind, key, content),
                )
                if kind == "measurement":
                    m = Measurement.model_validate(record)
                    self.db.execute(
                        "INSERT OR IGNORE INTO measurements VALUES(?,?,?,?,?)",
                        (key, m.model, m.target, m.created, m.comparison_key),
                    )
                if kind in {"measurement", "model-structure"}:
                    data = (
                        record.model_dump(mode="json") if hasattr(record, "model_dump") else record
                    )
                    changed[data["model"]] = data
            for record in changed.values():
                self._assimilate(record)

    def derive(self, model, *, cutoff=None):
        """Reproduce a model report solely from its stored evidence closure."""
        from formula_performance.evidence import evaluate
        from formula_performance.records import Publication

        if cutoff is None:
            cutoff = (
                self.db.execute("SELECT COALESCE(MAX(sequence),0) FROM records").fetchone()[0]
                if self.db is not None
                else 0
            )
        publications = []
        historical = 0
        for measured in self.records("measurement", cutoff=cutoff):
            if measured["model"] != model:
                continue
            evidence = measured.get("artifacts", {}).get("formula-performance")
            if evidence:
                publications.append(Publication.model_validate_json(self.blob(evidence)))
            elif measured["engine"] == "magnitude":
                historical += 1
        for structure in self.records("model-structure", cutoff=cutoff):
            if structure["model"] == model and structure.get("performance_manifest"):
                publications.append(Publication(manifests=(structure["performance_manifest"],)))
        report = {
            **evaluate(publications),
            "model": model,
            "historical_without_contract": historical,
        }
        return report

    def _assimilate(self, record):
        """Evidence and the resulting snapshot commit in the same transaction."""
        data = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        model = data["model"]
        report = self.derive(model)
        self.db.execute(
            "INSERT OR IGNORE INTO records(kind,id,content) VALUES('performance-snapshot',?,?)",
            (model + ":" + hashlib.sha256(encoded(report)).hexdigest(), encoded(report)),
        )
        self.db.execute(
            "INSERT INTO records(kind,id,content) VALUES('performance-model',?,?) "
            "ON CONFLICT(kind,id) DO UPDATE SET content=excluded.content",
            (model, encoded(report)),
        )

    def request(self, request_id):
        return Request.model_validate(self.get("request", request_id))

    def measurement(self, measurement_id):
        return Measurement.model_validate(self.get("measurement", measurement_id))

    def source(self, source_id):
        source = Source.model_validate(self.get("source", source_id))
        if source.source_id != source_id:
            raise ValueError("source manifest digest mismatch")
        return source

    def blob_path(self, identity):
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid artifact hash")
        return self.root / "blobs" / identity[:2] / identity[2:]

    def blob(self, identity):
        content = self.blob_path(identity).read_bytes()
        if hashlib.sha256(content).hexdigest() != identity:
            raise ValueError(f"corrupt artifact: {identity}")
        return content

    def put_blob(self, content):
        if self.readonly:
            raise RuntimeError("read-only evidence store")
        identity = hashlib.sha256(content).hexdigest()
        path = self.blob_path(identity)
        if not path.exists():
            atomic_write(path, content)
        elif self.blob(identity) != content:
            raise ValueError("artifact collision")
        return identity
