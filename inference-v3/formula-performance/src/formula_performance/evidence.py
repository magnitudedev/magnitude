"""Pure assimilation: all operating points share one formula performance relation."""

import json
from collections import defaultdict
from statistics import median

from .behavior import characterize
from .derive import contributions, formula_model, roofline
from .records import Publication, identity
from .revision import revision


def _merge(records, key):
    result = {}
    for record in records:
        k = key(record)
        if k in result and result[k] != record:
            raise ValueError("immutable analytical identity collision: " + k)
        result[k] = record
    return dict(sorted(result.items()))


def evaluate(publications):
    publications = [
        p if isinstance(p, Publication) else Publication.model_validate(p) for p in publications
    ]
    manifests = _merge((r for p in publications for r in p.manifests), identity)
    hardware = _merge((r for p in publications for r in p.hardware), identity)
    observations = _merge((r for p in publications for r in p.observations), lambda r: r.identity)
    captures = _merge((r for p in publications for r in p.captures), lambda r: r.identity)
    transfers = _merge((r for p in publications for r in p.transfers), lambda r: r.identity)
    # A native contribution becomes a throughput point only when the capture
    # covers the whole formula boundary without an indivisible crossing region.
    for observation in tuple(observations.values()):
        for capture_id in observation.captures:
            captured = captures[capture_id]
            manifest = manifests[captured.manifest]
            for component, contribution in contributions(manifest, captured).items():
                if contribution["complete_boundary"] and contribution["inclusive_seconds"] > 0:
                    key = capture_id + ":" + component
                    observations[key] = observation.model_copy(
                        update={
                            "identity": key,
                            "component": component,
                            "samples": (contribution["inclusive_seconds"],),
                            "boundary": "native-complete-formula-union",
                            "captures": (),
                        }
                    )
    relations = {}
    dependencies = defaultdict(set)
    # Preserve all graph realizations. A latest graph cannot erase another phase's evidence.
    for manifest_id, manifest in manifests.items():
        for formula in manifest.formulas:
            key = formula.component
            relation = relations.setdefault(
                key,
                {
                    "component": key,
                    "parent": formula.parent,
                    "label": formula.label,
                    "definition": formula.definition,
                    "metric": formula.primary.model_dump(),
                    "realizations": [],
                    "points": [],
                    "contributions": [],
                    "predictions": [],
                    "constraints": [],
                    "children": [],
                },
            )
            if relation["definition"] != formula.definition or relation["parent"] != formula.parent:
                raise ValueError("component mapping changes its formula/parent")
            if (
                relation["metric"]["unit"] != formula.primary.unit.model_dump()
                or relation["metric"]["name"] != formula.primary.name
            ):
                raise ValueError("formula metric changed without a component contract revision")
            relation["realizations"].append(
                {
                    "manifest": manifest_id,
                    "semantics": formula.semantics,
                    "definition_version": formula.version,
                    "model": formula_model(manifest, formula),
                }
            )
            if formula.parent is not None:
                dependencies[key].add(formula.parent)
    for key, ancestors in dependencies.items():
        for parent in ancestors:
            relations[parent]["children"].append(key)
    points = {}
    for observation in sorted(observations.values(), key=lambda o: (o.created, o.identity)):
        manifest = manifests[observation.manifest]
        formula = next(f for f in manifest.formulas if f.component == observation.component)
        system = hardware[observation.hardware]
        bound = roofline(manifest, formula, system)
        seconds = median(observation.samples) if observation.samples else None
        qualified = observation.correctness == "passed" and observation.status == "complete"
        rate = bound["quantity"] / seconds if seconds else None
        point = {
            "observation": observation.identity,
            "semantics": formula.semantics,
            "created": observation.created,
            "hardware": system.label,
            "hardware_binding": observation.hardware,
            "implementation": observation.implementation,
            "coordinates": observation.coordinates,
            "manifest": observation.manifest,
            "seconds": seconds,
            "rate": rate,
            "qualified": qualified,
            "correctness": observation.correctness,
            "status": observation.status,
            "boundary": observation.boundary,
            "roofline": bound,
            "attainment": rate / bound["ceiling"]
            if qualified and rate is not None and bound["ceiling"]
            else None,
            "evidence": observation.evidence,
        }
        if point["attainment"] is not None and point["attainment"] > 1:
            point["bound_status"] = "challenged"
        else:
            point["bound_status"] = bound["kind"]
        points[observation.identity] = point
        relations[observation.component]["points"].append(point)
        for term in bound["terms"]:
            dependencies["obligation:" + term["obligation"]].add(observation.component)
            for parameter in term["capacity_parameters"]:
                dependencies["capacity:" + observation.hardware + ":" + parameter].add(
                    observation.component
                )
        dependencies[observation.identity].add(observation.component)
        dependencies["implementation:" + observation.implementation].add(observation.component)
        dependencies["manifest:" + observation.manifest].add(observation.component)
    for observation in observations.values():
        relation = relations[observation.component]
        point = points[observation.identity]
        if not relation["children"] or not point["qualified"]:
            continue
        constraint = {
            "kind": "joint-total",
            "observation": observation.identity,
            "parent": observation.component,
            "children": sorted(relation["children"]),
            "remaining_seconds": point["seconds"],
            "boundary": observation.boundary,
            "assumptions": ("joint enclosing total; individual child costs are not identified",),
        }
        relation["constraints"].append(constraint)
        for child in relation["children"]:
            relations[child]["constraints"].append(constraint)
            dependencies[observation.identity].add(child)
    for capture in captures.values():
        for component, contribution in contributions(manifests[capture.manifest], capture).items():
            if contribution["regions"]:
                relations[component]["contributions"].append(contribution)
                dependencies[capture.identity].add(component)
                if contribution["parent_seconds"] is not None:
                    relations[component]["constraints"].append(
                        {
                            "kind": "same-capture-accounting",
                            "observation": capture.identity,
                            "remaining_seconds": contribution["parent_seconds"]
                            - contribution["inclusive_seconds"],
                            "known_seconds": contribution["inclusive_seconds"],
                            "assumptions": (
                                "native interval unions; shared and overlapping regions "
                                "are not additive",
                            ),
                        }
                    )
    from .propagation import observed_transfers, propagate

    automatic = observed_transfers(observations, manifests, captures)
    # Explicit user-authored transfer contracts take precedence for the same edge/baseline.
    declared = {(t.baseline_parent, t.child) for t in transfers.values()}
    transfers = {
        **{k: t for k, t in automatic.items() if (t.baseline_parent, t.child) not in declared},
        **transfers,
    }
    known_evidence = observations.keys() | captures.keys()
    if any(not set(t.evidence) <= known_evidence for t in transfers.values()):
        raise ValueError("transfer evidence closure is incomplete")
    for transfer in transfers.values():
        if transfer.relationship != "same-execution":
            continue
        proven = False
        for capture_id in transfer.evidence:
            captured = captures.get(capture_id)
            if captured is None or transfer.scale != 1:
                continue
            if (
                transfer.baseline_child != capture_id + ":" + transfer.child
                or transfer.baseline_parent != capture_id + ":" + transfer.parent
            ):
                continue
            accounting = contributions(manifests[captured.manifest], captured)
            child, parent = accounting.get(transfer.child), accounting.get(transfer.parent)
            if (
                not child
                or not parent
                or not child["complete_boundary"]
                or not parent["complete_boundary"]
            ):
                continue
            own = [r for r in captured.regions if r.identity in child["regions"]]
            rest = [
                r
                for r in captured.regions
                if r.identity in parent["regions"] and r.identity not in child["regions"]
            ]
            proven = transfer.contribution == child["inclusive_seconds"] and not any(
                a.start < b.end and b.start < a.end for a in own for b in rest
            )
            if proven:
                break
        if not proven:
            raise ValueError(
                "same-execution transfer requires an exact non-overlapping capture mapping"
            )
    propagate(relations, observations, points, transfers, dependencies)
    for relation in relations.values():
        relation["behavior"] = characterize(relation)
        values = [p["attainment"] for p in relation["points"] if p["attainment"] is not None]
        relation["attainment"] = (
            {"minimum": min(values), "maximum": max(values), "points": len(values)}
            if values
            else None
        )
        relation["evidence_count"] = len(relation["points"])
    return json.loads(
        json.dumps(
            {
                "version": 1,
                "analysis_revision": revision(),
                "watermark": identity((revision(), sorted({identity(p) for p in publications}))),
                "components": relations,
                "dependencies": {key: sorted(value) for key, value in sorted(dependencies.items())},
                "hardware": [h.model_dump() for h in hardware.values()],
                "observation_count": len(observations),
            }
        )
    )


def affected(report, changed):
    pending, seen = list(changed), set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        pending.extend(report["dependencies"].get(key, []))
    return sorted(seen & report["components"].keys())
