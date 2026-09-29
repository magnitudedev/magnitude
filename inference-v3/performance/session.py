"""Import verbatim session-bench evidence into the same assessment store.

The external HTTP composition stays opaque. Native service diagnostics remain
source evidence; they are not substituted for client-observed latency.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from performance.metrics import Latency, Record, TimingBoundary
from performance.store import Store
from session_bench.client import Observation
from session_bench.sessions import Request, Section


class SourceFile(Record):
    path: str
    sha256: str


class SessionRow(Record):
    target: str
    block: int = Field(ge=0)
    phase: Literal["warmup", "measured"]
    section: Section
    workload: Literal["tools", "prose", "retrieval"]
    checkpoint: int = Field(ge=0)
    concurrency: int = Field(ge=1)
    session: str
    fixture_id: str
    timing_basis: Literal["native-model-service", "server-token-emission"]
    observation: Observation
    retrieval_total: int | None = None


class SessionMetrics(Record):
    first_token_latency: Latency
    completed_latency: Latency


class ExternalSession(Record):
    thermals: JsonValue


class SessionAssessment(Record):
    kind: Literal["session_http"] = "session_http"
    source_directory: str
    source_files: tuple[SourceFile, ...]
    server_internals: Literal["opaque"] = "opaque"
    row: SessionRow
    request: Request
    metrics: SessionMetrics
    external: ExternalSession


class Ingestion(Record):
    imported: int
    duplicate: int


def ingest(directory: Path, store: Store) -> Ingestion:
    results = directory / "results.jsonl"
    if not results.exists():
        return Ingestion(imported=0, duplicate=0)
    header = json.loads((directory / "run.json").read_text())
    if "hardware" not in header.get("host", {}):
        raise ValueError("session evidence is missing hardware provenance")
    thermal_path = directory / "thermals.json"
    thermal = None
    if thermal_path.exists():
        thermal = json.loads(thermal_path.read_text())
        with (directory / "thermals.jsonl").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != thermal["trace_sha256"]:
                raise ValueError("session thermal trace checksum mismatch")
    requests = {
        request.id: request
        for line in (directory / "requests.jsonl").read_text().splitlines()
        if line
        for request in (Request.model_validate_json(line),)
    }
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append(SourceFile(path=str(path.relative_to(directory)), sha256=checksum))
    imported = duplicate = 0
    for line in results.read_text().splitlines():
        if not line:
            continue
        row = SessionRow.model_validate_json(line)
        if row.phase != "measured":
            continue
        observation = row.observation
        record = SessionAssessment(
            external=ExternalSession(thermals=thermal),
            source_directory=str(directory),
            source_files=tuple(files),
            row=row,
            request=requests[observation.request_id],
            metrics=SessionMetrics(
                first_token_latency=Latency(
                    samples_seconds=()
                    if observation.ttft_ms is None
                    else (observation.ttft_ms / 1000,),
                    boundary=TimingBoundary.HTTP_FIRST_TOKEN,
                    missing_model_evidence=("opaque HTTP server; no applicable lower-time model",),
                ),
                completed_latency=Latency(
                    samples_seconds=()
                    if observation.completed_ms is None
                    else (observation.completed_ms / 1000,),
                    boundary=TimingBoundary.HTTP_COMPLETION,
                    missing_model_evidence=("opaque HTTP server; no applicable lower-time model",),
                ),
            ),
        )
        _, created = store.put(record)
        imported += created
        duplicate += not created
    return Ingestion(imported=imported, duplicate=duplicate)
