"""Import the existing serving benchmark's evidence without changing its client.

HTTP observations describe the actual session-bench client/server composition. The
server stays opaque unless it exported an actual component graph. Native model
service timings are retained as evidence and never substituted for HTTP durations.
"""

import hashlib
import json
from pathlib import Path

from magnitude_engine.components import component_id
from performance.facts import Configuration
from performance.records import Assembly, Node, Profile, digest
from performance.store import Store
from session_bench.client import measure


def ingest(directory: Path, store: Store) -> dict:
    directory = Path(directory)
    header = json.loads((directory / "run.json").read_text())
    if "hardware" not in header.get("host", {}):
        raise ValueError("session result lacks hardware provenance; retain as historical evidence")
    thermal_path = directory / "thermals.json"
    thermals = json.loads(thermal_path.read_text()) if thermal_path.exists() else None
    if thermals is not None:
        with (directory / "thermals.jsonl").open("rb") as trace:
            if hashlib.file_digest(trace, "sha256").hexdigest() != thermals["trace_sha256"]:
                raise ValueError("session temperature trace checksum mismatch")
    results = directory / "results.jsonl"
    if not results.exists():
        return {"imported": 0, "duplicate": 0}
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    requests = {
        r["id"]: r
        for r in (
            json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()
        )
    }
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    duplicate = 0
    pending = []
    for row in rows:
        if row["phase"] != "measured":
            continue
        target = row["target"]
        observation = row["observation"]
        request = requests[observation["request_id"]]
        artifact = json.loads((directory / f"{target}-artifact.json").read_text())
        runtime = json.loads((directory / f"{target}-runtime.json").read_text())
        source = {
            "session": header["id"],
            "row": row,
            "request": request,
            "artifact": artifact,
            "runtime": runtime,
            "thermals": thermals,
        }
        checksum = digest(source)
        identity = digest({"session_bridge": 2, "source": checksum})[:32]
        if (store.root / "runs" / identity / "run.json").exists():
            duplicate += 1
            continue
        # Host-local artifact paths and source checkout paths are provenance only.
        artifact_key = {
            "kind": artifact["kind"],
            "files": [
                {"path": f["path"], "size": f["size"], "sha256": f["sha256"]}
                for f in artifact["files"]
            ],
            "metadata": artifact["metadata"],
        }
        implementation = Node(
            component_id(measure),
            digest(runtime.get("files", runtime)),
            Configuration(
                settings={
                    "server": target.rsplit("-", 1)[0],
                    "opaque_server": True,
                    "artifact": digest(artifact_key),
                }
            ),
        )
        graph = Assembly(
            "service", {"service": implementation}, f"{target} · HTTP", {"target": artifact_key}
        )
        profile = Profile(
            header["host"]["hardware"],
            {"server": runtime.get("dependencies", {}), "client_python": header["host"]["python"]},
        )
        terminal = observation.get("terminal") or {}
        metrics = {}
        if observation.get("ttft_ms") is not None:
            metrics["TTFT"] = observation["ttft_ms"] / 1000
        completed = observation.get("completed_ms")
        usage = terminal.get("usage", {})
        output_tokens = usage.get("completion_tokens")
        if completed and output_tokens is not None:
            metrics["RATE"] = output_tokens / (completed / 1000)
        workload = {
            "request": {k: v for k, v in request.items() if k not in ("id", "session")},
            "concurrency": row["concurrency"],
            "timing_basis": row["timing_basis"],
            "statistics": {"TTFT": "single-request"},
        }
        status = "complete" if observation["outcome"] == "valid" else "failed"
        completion = next(
            (
                e["at"]
                for e in events
                if e.get("event") == "request_finished"
                and e.get("target") == target
                and e.get("request") == request["id"]
                and e.get("block") == row["block"]
                and e.get("phase") == "measured"
            ),
            None,
        )
        if completion is None:
            raise ValueError("session observation has no recorded completion event")
        record = {
            "schema_version": 2,
            "id": identity,
            "status": status,
            "started_at": header["started_at"],
            "completed_at": completion,
            "benchmark": "session.http",
            "boundary": "http-request-through-terminal-event",
            "node": graph.root,
            "assembly": graph.record(),
            "profile": profile.record(),
            "workload": workload,
            "warmup": 0,
            "repetitions": 1,
            "timing_dimension": None,
            "samples": [
                {
                    "phase": "measured",
                    "index": 0,
                    "elapsed_ns": int((completed or 0) * 1000000),
                    "observation": {
                        "output_digest": digest(observation),
                        "metrics": metrics,
                        "evidence": observation,
                    },
                }
            ],
            "external": {
                "directory": str(directory),
                "checksum": checksum,
                **source,
                "warmups": [
                    r
                    for r in rows
                    if r["target"] == target
                    and r["block"] == row["block"]
                    and r["phase"] == "warmup"
                ],
            },
        }
        record["checksum"] = digest(record)
        pending.append(record)
    result = store.ingest_many(pending)
    result["duplicate"] += duplicate
    return result
