"""Shared read-only reports for agents and the Textual browser."""

import base64
import json
import statistics

from .contracts import Measurement, digest, encoded

PAGE = 24


def timing(m):
    values = m.samples_seconds
    return {
        "median_seconds": statistics.median(values) if values else None,
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
        "sample_count": len(values),
    }


def summary(m):
    return {
        "measurement_id": m.measurement_id,
        "created": m.created,
        "source_id": m.source_id,
        "target": m.target,
        "engine": m.engine,
        "scope": m.scope,
        "status": m.status,
        "correctness": m.correctness,
        **timing(m),
        "error": m.error[:1024] if m.error else None,
        "unavailable": m.unavailable,
    }


def analysis(m):
    """Interpret the recorded resource model without loading its execution runtime."""
    result = dict(m.details.get("analysis") or {})
    roofline = result.get("roofline")
    if roofline and roofline.get("revision") == "formula-resource-roofline-v2":
        pools = {}
        for limit in roofline["limits"]:
            demand = limit["demand"]
            key = (demand["resource"], digest(demand.get("source")))
            pools[key] = pools.get(key, 0) + demand["lower"] / limit["rate"]
        floor = max(pools.values(), default=0)
        duration = timing(m)["median_seconds"]
        result["modeled_seconds"] = floor
        result["limiting_resource"] = max(pools, key=pools.get)[0] if pools else None
        result["modeled_percent_of_measured"] = 100 * floor / duration if duration else None
        result["basis"] = "formula demand / empirical resource rates; ideal overlap and reuse"
    return result or None


def compare(a, b):
    differences = {
        key: {"baseline": getattr(a, key), "candidate": getattr(b, key)}
        for key in ("artifact", "workload", "scope", "engine", "hardware", "protocol")
        if getattr(a, key) != getattr(b, key)
    }
    if a.details.get("comparison") != b.details.get("comparison"):
        differences["component_contract"] = {
            "baseline": a.details.get("comparison"),
            "candidate": b.details.get("comparison"),
        }
    if any(
        m.scope.startswith(("decode/", "prefill/")) and not m.details.get("comparison")
        for m in (a, b)
    ):
        differences["component_contract"] = "logical input correspondence unavailable"
    compatible = a.comparison_key == b.comparison_key and not differences
    qualified = compatible and all(
        m.status == "complete" and m.correctness == "passed" for m in (a, b)
    )
    paired = bool(qualified and a.pair_ids and a.pair_ids == b.pair_ids)
    left, right = timing(a)["median_seconds"], timing(b)["median_seconds"]
    result = {
        "baseline": summary(a),
        "candidate": summary(b),
        "differences": differences,
        "compatible": compatible,
        "qualified": qualified,
        "evidence": "paired" if paired else "historical",
        "delta_seconds": right - left
        if compatible and left is not None and right is not None
        else None,
        "latency_change_percent": 100 * (right / left - 1)
        if compatible and left and right is not None
        else None,
    }
    if paired:
        deltas = [y - x for x, y in zip(a.samples_seconds, b.samples_seconds, strict=True)]
        result["paired_deltas_seconds"] = deltas
    # Descriptive spread, not an invented statistical significance test.
    result["conclusion"] = (
        "incompatible conditions"
        if not compatible
        else "numerically unqualified"
        if not qualified
        else "overlapping sample ranges; change is inconclusive"
        if max(a.samples_seconds) >= min(b.samples_seconds)
        and max(b.samples_seconds) >= min(a.samples_seconds)
        else "non-overlapping observed ranges; bounded sample evidence"
    )
    return result


def _cursor(value):
    return base64.urlsafe_b64encode(encoded(value)).decode()


def _read_cursor(value, selection):
    try:
        data = json.loads(base64.urlsafe_b64decode(value))
        if data["selection"] != digest(selection) or data["offset"] < 0:
            raise ValueError("cursor belongs to another query")
        return data
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("invalid query cursor") from exc


class Queries:
    def __init__(self, store, models=None, targets=None):
        self.store = store
        self.models = models or {}
        self.targets = targets or {}

    def tree(self, model):
        from .model_view import build_tree

        return build_tree(self.store, model, self.models.get(model))

    def model(self, model, *, filters=None, cursor=None):
        filters = filters or {}
        selection = {"model": model, "filters": filters}
        cutoff = (
            self.store.db.execute("SELECT COALESCE(MAX(sequence),0) FROM records").fetchone()[0]
            if self.store.db
            else 0
        )
        page = (
            _read_cursor(cursor, selection)
            if cursor
            else {"selection": digest(selection), "offset": 0, "cutoff": cutoff}
        )
        records = (
            self.store.db.execute(
                "SELECT r.content FROM records r JOIN measurements m ON m.id=r.id "
                "WHERE r.kind='measurement' AND m.model=? AND r.sequence<=? "
                "ORDER BY m.created DESC,m.id",
                (model, page["cutoff"]),
            ).fetchall()
            if self.store.db
            else []
        )
        measurements = [Measurement.model_validate_json(row[0]) for row in records]

        def matches(m):
            for key, value in filters.items():
                if value is None:
                    continue
                actual = (
                    m.workload.get(key)
                    if key in ("workload", "context", "steps", "step")
                    else getattr(m, key, None)
                )
                if key == "targets":
                    if m.target not in value:
                        return False
                elif key == "scope":
                    if (
                        m.scope != value
                        and not m.scope.startswith(value + "/")
                        and not any(s.selector == value for s in m.scopes)
                    ):
                        return False
                elif actual != value:
                    return False
            return True

        all_measurements = measurements
        measurements = [m for m in measurements if matches(m)]
        groups = {}
        for m in measurements:
            # Hardware and resolved input contracts define groups, never the display alias alone.
            groups.setdefault(m.comparison_key, []).append(m)
        values = []
        for key, history in groups.items():
            latest = history[0]
            valid = [m for m in history if m.correctness == "passed" and m.status == "complete"]
            best = min(valid, key=lambda m: statistics.median(m.samples_seconds)) if valid else None
            values.append(
                {
                    "condition_id": key,
                    "workload": latest.workload,
                    "hardware": latest.hardware,
                    "artifact": latest.artifact,
                    "model_definition": self.definition(latest),
                    "enclosing": [
                        summary(m)
                        for m in measurements
                        if latest.scope.startswith(m.scope + "/")
                        and m.scope in {"prefill", "decode"}
                        and m.source_id == latest.source_id
                        and m.artifact == latest.artifact
                        and m.hardware == latest.hardware
                        and m.engine == latest.engine
                        and m.workload.get("tokens") is not None
                        and m.workload.get("tokens") == latest.workload.get("tokens")
                    ][:8],
                    "latest": summary(latest),
                    "previous": summary(history[1]) if len(history) > 1 else None,
                    "best_correct": summary(best) if best else None,
                    "change": compare(history[1], latest) if len(history) > 1 else None,
                    "measurement_count": len(history),
                    "recent": [summary(m) for m in history[:8]],
                    "scopes": [s.model_dump() for s in latest.scopes[:PAGE]],
                    "scope_count": len(latest.scopes),
                    "selected_scope": next(
                        (
                            s.model_dump()
                            for s in latest.scopes
                            if s.selector == filters.get("scope")
                        ),
                        None,
                    ),
                    "analysis": analysis(latest),
                    "references": [
                        {
                            **summary(m),
                            "boundary": m.details.get("boundary") or m.protocol.get("boundary"),
                            "sampling": m.details.get("sampling"),
                            "correspondence": (
                                "same artifact and tokens; engine timer boundaries differ"
                            ),
                        }
                        for m in all_measurements
                        if m.engine != latest.engine
                        and m.artifact == latest.artifact
                        and m.scope == latest.scope.split("/")[0]
                        and m.workload.get("tokens") is not None
                        and m.workload.get("tokens") == latest.workload.get("tokens")
                    ][:8],
                    "artifacts": latest.artifacts,
                    "costs": latest.costs,
                }
            )
        start = page["offset"]
        following = {**page, "offset": start + PAGE}
        configured = self.models.get(model)
        if (
            configured is None
            and not all_measurements
            and not any(
                s["model"] == model
                for s in self.store.records("model-structure", cutoff=page["cutoff"])
            )
        ):
            raise KeyError(f"unknown model: {model}")
        present = {m.target for m in measurements}
        performance = self.store.derive(model, cutoff=page["cutoff"])
        scope = filters.get("scope")
        if scope and performance["components"]:
            components = performance["components"]
            selected = scope
            if selected not in components:
                # Execution selectors include phase and the numerical root.
                # Resolve those declared prefixes; the report remains model-wide.
                parts = scope.split("/")
                roots = [r["definition"] for r in components.values() if r["parent"] is None]
                if parts[0] in {"decode", "prefill"}:
                    parts = parts[1:]
                    if parts and parts[0] in {r + "[0]" for r in roots}:
                        parts = parts[1:]
                    selected = "/".join(parts)
            if selected not in components:
                raise KeyError(f"unknown model component: {scope}")
            keep = {selected, *components[selected]["children"]}
            parent = components[selected]["parent"]
            while parent is not None:
                keep.add(parent)
                parent = components[parent]["parent"]
            performance = {
                **performance,
                "selected_component": selected,
                "components": {k: v for k, v in components.items() if k in keep},
            }
        return {
            "performance": performance,
            "model": model,
            "definition": configured.model_dump() if configured else None,
            "measurement_count": len(measurements),
            "condition_count": len(values),
            "groups": values[start : start + PAGE],
            "missing_targets": [
                name for name in (configured.locations if configured else ()) if name not in present
            ],
            "current_source": "unknown until measured or explicitly verified",
            "cursor": _cursor(following) if start + PAGE < len(values) else None,
        }

    def definition(self, measurement):
        if measurement.details.get("model_definition"):
            return measurement.details["model_definition"]
        try:
            request = self.store.get("request", measurement.request_id)
        except KeyError:
            return None
        return request.get("model")

    def source(self, source_id, *, cursor=None):
        source = self.store.source(source_id)
        selection = {"source": source_id}
        page = (
            _read_cursor(cursor, selection)
            if cursor
            else {"selection": digest(selection), "offset": 0}
        )
        start, size = page["offset"], 100
        return {
            "source_id": source_id,
            "file_count": len(source.files),
            "files": [f.model_dump() for f in source.files[start : start + size]],
            "cursor": _cursor({**page, "offset": start + size})
            if start + size < len(source.files)
            else None,
        }

    def measurement(self, measurement_id, *, cursor=None):
        m = self.store.measurement(measurement_id)
        selection = {"measurement": measurement_id}
        page = (
            _read_cursor(cursor, selection)
            if cursor
            else {"selection": digest(selection), "offset": 0}
        )
        start = page["offset"]
        data = m.model_dump(mode="json")
        data["scopes"] = data["scopes"][start : start + PAGE]
        # Large engine-native detail is separately addressable as an artifact.
        data["scope_count"] = len(m.scopes)
        data["error"] = m.error[:2048] if m.error else None
        data["timing"] = timing(m)
        data["analysis"] = analysis(m)
        data["cursor"] = (
            _cursor({**page, "offset": start + PAGE}) if start + PAGE < len(m.scopes) else None
        )
        return data

    def artifact(self, artifact_id, *, cursor=None):
        selection = {"artifact": artifact_id}
        page = (
            _read_cursor(cursor, selection)
            if cursor
            else {"selection": digest(selection), "offset": 0}
        )
        content = self.store.blob(artifact_id)
        start, size = page["offset"], 16384
        # Base64 preserves exact bytes, including UTF-8 codepoints split at a page edge.
        chunk = content[start : start + size]
        try:
            value, encoding = chunk.decode(), "utf-8"
        except UnicodeDecodeError:
            value, encoding = base64.b64encode(chunk).decode(), "base64"
        owners = [
            r["measurement_id"]
            for r in self.store.records("measurement")
            if artifact_id in r["artifacts"].values()
        ]
        return {
            "artifact_id": artifact_id,
            "size_bytes": len(content),
            "offset": start,
            "content": value,
            "encoding": encoding,
            "measurements": owners[:PAGE],
            "measurement_count": len(owners),
            "requests": [
                r["request_id"]
                for r in self.store.records("request")
                if artifact_id in r.get("inputs", {}).values()
                or any(artifact_id in a.get("artifact_ids", ()) for a in r["attempts"])
            ][:PAGE],
            "cursor": _cursor({**page, "offset": start + size})
            if start + size < len(content)
            else None,
        }
