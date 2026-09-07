"""Deterministic evidence joins and two evaluations of captured component graphs."""

from __future__ import annotations

import statistics
from dataclasses import asdict

from performance.records import Assembly, Profile, digest
from performance.theory.catalog import DIMENSIONS, evaluate, requirements, revision
from performance.theory.engine import neural_point
from performance.theory.resources import Bound, Demands


def point_for(workload: dict, path: str) -> dict:
    return {k: v for k, v in workload.items() if k != "nodes"} | workload.get("nodes", {}).get(
        path, {}
    )


def evidence_key(component: str, profile: str, point: dict, boundary: str) -> str:
    return digest(
        {"component": component, "profile": profile, "point": point, "boundary": boundary}
    )


def observation_binding(run: dict, path: str) -> tuple[dict, str]:
    """Explicit per-occurrence bindings connect unlike parent/child boundaries.

    An override supplies the complete child point; it is never silently mixed
    with a containing engine workload. Absent overrides require exact matching.
    """
    selected = run.get("bindings", {}).get(path, {})
    point = selected.get("workload", point_for(run["workload"], path))
    contract = digest(
        {
            "boundary": selected.get("boundary", run["boundary"]),
            "version": selected.get("contract_version", run.get("contract_version", "1")),
            "statistic": "median-of-repetitions",
        }
    )
    return point, contract


def formulate(graph: Assembly, workload: dict, profile: Profile, *, bindings=None) -> dict:
    """Evaluate every component without loading a model or collecting a sample."""
    points, demands, bounds = {}, {}, {}

    def visit(path):
        if path in bounds:
            return
        node = graph.nodes[path]
        point = points[path] = (
            (bindings or {}).get(path, {}).get("workload", point_for(workload, path))
        )
        for child in (*node.children.values(), *node.dependencies.values()):
            visit(child)
        inputs = neural_point(point) | {"information_domain": path}
        if "kv_source" in node.parameters:
            inputs["kv_information_domain"] = (
                path.split(".layers.")[0] + ".kv." + str(node.parameters["kv_source"])
            )
        try:
            demand = requirements(
                node, inputs, {name: demands[p] for name, p in node.children.items()}
            )
            result = evaluate(
                node, point, profile, demand, {name: bounds[p] for name, p in node.children.items()}
            )
        except (ValueError, KeyError, TypeError) as error:
            # Bad input remains visible; it cannot prevent raw failure persistence.
            from performance.theory.catalog import METRICS

            missing = (f"invalid binding: {error}",)
            demand = Demands(missing=missing)
            result = {
                d: Bound(None, METRICS[d]["unit"], missing=missing)
                for d in DIMENSIONS.get(node.component, ())
            }
            if not result:
                raise ValueError(f"undefined component contract: {node.component}") from error
        demands[path], bounds[path] = demand, result

    visit(graph.root)
    return {"points": points, "demands": demands, "bounds": bounds}


def preflight(graph: Assembly, workload: dict, profile: Profile, *, bindings=None) -> dict:
    """Serializable formulation audit to resolve before a percentage campaign."""
    result = formulate(graph, workload, profile, bindings=bindings)
    return {
        path: {d: asdict(b) for d, b in dimensions.items()}
        for path, dimensions in result["bounds"].items()
    }


def rebuild(runs: list[dict]) -> dict:
    ordered = sorted(runs, key=lambda r: (r.get("completed_at", r["started_at"]), r["id"]))
    state = {
        "schema_version": 1,
        "theory_revision": revision(),
        "compositions": {},
        "profiles": {},
        "components": {},
        "views": {},
        "runs": {},
    }
    observations: dict[str, dict] = {}
    for run in ordered:
        graph = Assembly.read(run["assembly"])
        profile = Profile(**run["profile"])
        state["profiles"][profile.identity] = profile.record()
        composition = state["compositions"].setdefault(
            graph.identity, {"label": graph.label, "revisions": {}}
        )
        composition["revisions"][graph.revision] = graph.record()
        composition["current_revision"] = graph.revision
        state["runs"][run["id"]] = {
            "status": run["status"],
            "benchmark": run["benchmark"],
            "completed_at": run.get("completed_at"),
            "composition": graph.identity,
        }
        if run["status"] != "complete":
            continue
        samples = [s for s in run["samples"] if s["phase"] == "measured" and not s.get("error")]
        if not samples:
            continue
        path = run["node"]
        key = evidence_key(
            graph.component_keys()[path],
            profile.identity,
            *observation_binding(run, path),
        )
        metrics = {}
        if run.get("timing_dimension", "EXEC") is not None:
            metrics[run.get("timing_dimension", "EXEC")] = (
                statistics.median(s["elapsed_ns"] for s in samples) / 1e9
            )
        names = set.intersection(*(set(s["observation"].get("metrics", {})) for s in samples))
        metrics.update(
            {
                name: statistics.median(s["observation"]["metrics"][name] for s in samples)
                for name in names
            }
        )
        for dimension, value in metrics.items():
            observations.setdefault(key, {})[dimension] = {
                "value": value,
                "evidence": [run["id"]],
                "benchmark": run["benchmark"],
                "estimated": False,
            }

    for run in ordered:
        assess_view(run, observations, state)
    publish_current_assessments(state)
    state["generation"] = digest(state)
    return state


def publish_current_assessments(state: dict) -> None:
    """Browse the latest evidence per dimension, retaining each actual operating point.

    This is a selection across points, never a pooled or hardware-normalized score.
    Parent assessments still come from their own measured/composed operating point.
    """
    candidates = {}
    for key, component in state["components"].items():
        candidates.setdefault(component["fingerprint"], []).append(key)

    def recency(value):
        return max(
            ((state["runs"][r]["completed_at"] or "", r) for r in value["evidence"]),
            default=("", ""),
        )

    for composition in state["compositions"].values():
        graph = Assembly.read(composition["revisions"][composition["current_revision"]])
        current = composition["current_assessments"] = {}
        for path, fingerprint in graph.component_keys().items():
            current[path] = {}
            for dimension in DIMENSIONS[graph.nodes[path].component]:
                keys = candidates.get(fingerprint, [])
                if keys:
                    current[path][dimension] = max(
                        keys,
                        key=lambda k: (
                            state["components"][k]["dimensions"][dimension]["observed"] is not None,
                            recency(state["components"][k]["dimensions"][dimension]),
                            k,
                        ),
                    )


def assess_view(run: dict, observations: dict, state: dict) -> None:
    graph, profile = Assembly.read(run["assembly"]), Profile(**run["profile"])
    fingerprints = graph.component_keys()
    formulation = formulate(graph, run["workload"], profile, bindings=run.get("bindings"))
    points, bounds, assessments = formulation["points"], formulation["bounds"], {}

    def visit(path):
        if path in assessments:
            return
        node = graph.nodes[path]
        point = points[path]
        for child in (*node.children.values(), *node.dependencies.values()):
            visit(child)
        key = evidence_key(fingerprints[path], profile.identity, *observation_binding(run, path))
        observation = observations.get(key)
        if node.component == "STATE:QWEN35" and "MEM" not in (observation or {}):
            # This contract owns disjoint KV-arena and recurrent-image backing.
            # Repeated references to the same child remain a single allocation.
            memory = [
                assessments[p]["MEM"]
                for p in set(node.children.values())
                if "MEM" in assessments[p]
            ]
            if memory and all(value["observed"] is not None for value in memory):
                observation = {
                    **(observation or {}),
                    "MEM": {
                        "value": sum(value["observed"] for value in memory),
                        "evidence": sorted({r for value in memory for r in value["evidence"]}),
                        "estimated": True,
                        "benchmark": "composed",
                    },
                }
        if observation is None and node.execution != "joint" and node.children:
            costs = [assessments[p].get("EXEC", {}) for p in node.children.values()]
            if all(c.get("observed") is not None for c in costs):
                multiplicities = point.get("invocations", {})
                values = [
                    c["observed"] * multiplicities.get(name, 1)
                    for name, c in zip(node.children, costs, strict=True)
                ]
                observation = {
                    "EXEC": {
                        "value": sum(values) if node.execution == "serial" else max(values),
                        "evidence": sorted({r for c in costs for r in c["evidence"]}),
                        "estimated": True,
                        "benchmark": "composed",
                    }
                }
        assessed = {}
        for dimension in DIMENSIONS[node.component]:
            bound = bounds[path][dimension]
            metric = (observation or {}).get(dimension, {})
            value = metric.get("value")
            reason = None
            percent = None
            if value is None:
                reason = "missing matched observation"
            elif bound.value is None:
                reason = (
                    "unbounded theoretical rate"
                    if bound.kind == "unbounded"
                    else "missing theoretical bindings"
                )
            elif bound.value <= 0:
                reason = "zero theoretical bound"
            elif value <= 0:
                reason = "nonpositive observed metric"
            else:
                percent = 100 * (
                    bound.value / value if bound.direction == "lower" else value / bound.value
                )
                if percent > 100 + 1e-9:
                    reason = "observation exceeds theoretical bound"
            assessed[dimension] = {
                "dimension": f"{node.component}/{dimension}",
                "observed": value,
                "bound": asdict(bound),
                "percent": percent,
                "issue": reason,
                "estimated": metric.get("estimated", False),
                "evidence": metric.get("evidence", []),
                "benchmark": metric.get("benchmark"),
                "formula_revision": state["theory_revision"],
            }
        assessments[path] = assessed
        state["components"][key] = {
            "implementation": node.implementation,
            "fingerprint": fingerprints[path],
            "profile": profile.identity,
            "workload": point,
            "dimensions": assessed,
        }

    visit(graph.root)
    view_key = digest(
        {
            "composition": graph.identity,
            "revision": graph.revision,
            "profile": profile.identity,
            "workload": run["workload"],
            "boundary": run["boundary"],
            "bindings": run.get("bindings", {}),
            "contract_version": run.get("contract_version", "1"),
        }
    )
    state["views"][view_key] = {
        "composition": graph.identity,
        "revision": graph.revision,
        "profile": profile.identity,
        "workload": run["workload"],
        "boundary": run["boundary"],
        "assessments": {
            path: evidence_key(
                fingerprints[path], profile.identity, *observation_binding(run, path)
            )
            for path in assessments
        },
    }
