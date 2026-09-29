"""Read session-bench's recorded interchange evidence without importing its runner."""

import json
import math
import re

from .contracts import Measurement, digest, encoded


def import_session(store, config, directory):
    files = sorted(p for p in directory.rglob("*") if p.suffix in {".json", ".jsonl"})
    if not files:
        raise ValueError("session directory has no JSON evidence")
    if any(p.is_symlink() or not p.resolve().is_relative_to(directory.resolve()) for p in files):
        raise ValueError("session evidence must be contained files")
    content = {p.relative_to(directory).as_posix(): p.read_bytes() for p in files}
    artifacts = {name: store.put_blob(value) for name, value in content.items()}
    session_id = digest(artifacts)

    def document(name, default=None):
        return json.loads(content[name]) if name in content else default

    def rows(name):
        return [json.loads(line) for line in content.get(name, b"").splitlines() if line.strip()]

    run, plan = document("run.json", {}), document("plan.json", {})
    requests = {r["id"]: r for r in rows("requests.jsonl")}
    events = {
        (e.get("target"), e.get("request"), e.get("block"), e.get("phase")): e["at"]
        for e in rows("events.jsonl")
        if e.get("event") == "request_finished"
    }
    imported, unresolved, measurements = [], [], []
    for index, row in enumerate(rows("results.jsonl")):
        if row.get("phase") != "measured":
            continue
        target = row["target"]
        artifact = document(target + "-artifact.json", {})
        entries = artifact.get("files", [])
        checksum = (
            entries[0].get("sha256")
            if artifact.get("kind") == "gguf" and len(entries) == 1
            else None
        )
        matches = [name for name, model in config.models.items() if model.sha256 == checksum]
        if len(matches) != 1:
            unresolved.append(
                {
                    "row": index,
                    "target": target,
                    "reason": "artifact checksum has no unique configured model",
                    "models": matches,
                }
            )
            continue
        if not re.fullmatch("[a-f0-9]{64}", checksum or ""):
            unresolved.append(
                {"row": index, "target": target, "reason": "missing artifact checksum"}
            )
            continue
        observation = row["observation"]
        request = requests.get(observation.get("request_id"), {})
        created = events.get(
            (target, observation.get("request_id"), row.get("block"), row.get("phase")),
            run.get("started_at"),
        )
        if run.get("format") != 1 or not created or not request or not row.get("timing_basis"):
            unresolved.append(
                {
                    "row": index,
                    "target": target,
                    "reason": "missing request, chronology or timer contract",
                }
            )
            continue
        definitions = [
            t
            for t in run.get("selection", {}).get("targets", [])
            if t.get("reference") == artifact.get("reference")
        ]
        engines = {
            t["engine"]
            for t in definitions
            if target == t["engine"] + "-" + digest(t["reference"])[:10]
        }
        if len(engines) != 1:
            unresolved.append(
                {"row": index, "target": target, "reason": "unresolved engine identity"}
            )
            continue
        native = (observation.get("terminal") or {}).get("timings", {})
        metrics = [
            ("request/first-token", observation.get("ttft_ms"), None, "http-first-token-latency"),
            ("request/completed", observation.get("completed_ms"), None, "http-completed-latency"),
            (
                "prefill",
                native.get("prompt_ms"),
                native.get("prompt_n"),
                row["timing_basis"] + ":prompt_ms",
            ),
            (
                "decode",
                native.get("predicted_ms"),
                native.get("predicted_n"),
                row["timing_basis"] + ":predicted_ms",
            ),
        ]
        for scope, milliseconds, count, boundary in metrics:
            if milliseconds is None:
                continue
            if (
                not isinstance(milliseconds, (float, int))
                or not math.isfinite(milliseconds)
                or milliseconds < 0
            ):
                raise ValueError(f"invalid session timer at row {index}: {scope}")
            model = matches[0]
            measurement_id = digest(
                {"session": session_id, "row": index, "scope": scope, "model": model}
            )
            m = Measurement(
                measurement_id=measurement_id,
                request_id=session_id,
                attempt_id=digest([session_id, target, row.get("block")]),
                created=created,
                model=model,
                artifact=checksum,
                source_id=None,
                target=target,
                engine=next(iter(engines)),
                scope=scope,
                workload={
                    "workload": row.get("workload"),
                    "section": row.get("section"),
                    "request": digest(request),
                    "context": request.get("context"),
                    "checkpoint": row.get("checkpoint"),
                    "concurrency": row.get("concurrency"),
                    "cache_policy": plan.get("cache_policy"),
                    "count": count,
                },
                hardware=run.get("host", {}).get("hardware", {}),
                protocol={
                    "boundary": boundary,
                    "version": "session-import-v1",
                    "input_identity": "HTTP request; rendered tokens unverified",
                },
                status="complete"
                if observation.get("outcome") in {"valid", "invalid"}
                else "incomplete",
                correctness="unchecked",
                samples_seconds=(milliseconds / 1000,),
                artifacts=artifacts,
                details={
                    "origin": "session-bench",
                    "session_id": session_id,
                    "result_row": index,
                    "boundary": boundary,
                    "session_outcome": observation.get("outcome"),
                    "model_definition": {
                        "sha256": checksum,
                        "locations": {target: artifact["path"]} if artifact.get("path") else {},
                    },
                    "engine_identity_artifact": artifacts.get(target + "-runtime.json"),
                },
                unavailable=(
                    "Independent numerical checking unavailable; response validation is separate",
                    "Formula attribution and replayable source unavailable",
                    "HTTP inputs do not establish rendered token/state equivalence",
                ),
                error=observation.get("error"),
            )
            measurements.append(m)
            imported.append(measurement_id)
    for measurement in measurements:
        store.put("measurement", measurement.measurement_id, measurement)
    report = {
        "session_id": session_id,
        "artifacts": artifacts,
        "measurement_ids": imported,
        "unresolved": unresolved,
    }
    artifact_id = store.put_blob(encoded(report))
    store.put("session-import", artifact_id, report)
    return {
        "artifact_id": artifact_id,
        "imported": len(imported),
        "measurement_ids": imported,
        "unresolved": unresolved,
    }
