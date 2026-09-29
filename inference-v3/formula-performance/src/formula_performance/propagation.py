"""Apply declared serial transfer relations in containment order."""

from collections import defaultdict


def propagate(relations, observations, points, transfers, dependencies):
    def depth(component):
        count = 0
        while relations[component]["parent"] is not None:
            component = relations[component]["parent"]
            count += 1
        return count

    grouped = defaultdict(list)
    for transfer in transfers.values():
        if transfer.baseline_child not in points or transfer.baseline_parent not in points:
            raise ValueError("transfer evidence closure is incomplete")
        child = observations[transfer.baseline_child]
        parent = observations[transfer.baseline_parent]
        if child.component != transfer.child or parent.component != transfer.parent:
            raise ValueError("transfer formula mismatch")
        if relations[transfer.child]["parent"] != transfer.parent:
            raise ValueError("a transfer must follow a declared direct composition edge")
        if transfer.relationship == "conditional-serial" and not transfer.assumptions:
            raise ValueError("conditional transfer needs its applicability assumptions")
        if child.hardware != parent.hardware:
            raise ValueError("serial transfer baselines must have the same hardware binding")
        if not points[child.identity]["qualified"] or not points[parent.identity]["qualified"]:
            continue
        grouped[parent.identity].append(transfer)

    for baseline_id, group in sorted(
        grouped.items(), key=lambda item: -depth(observations[item[0]].component)
    ):
        parent = observations[baseline_id]
        parent_point = points[baseline_id]
        relation = relations[parent.component]
        if sum(t.contribution for t in group) > parent_point["seconds"]:
            raise ValueError("serial contributions exceed the parent boundary")
        if len({t.child for t in group}) != len(group):
            raise ValueError("overlapping serial transfer contributions")
        candidates = defaultdict(list)
        for transfer in group:
            child = observations[transfer.baseline_child]
            baseline = points[child.identity]
            child_relation = relations[transfer.child]
            for candidate in (*child_relation["points"], *child_relation["predictions"]):
                if (
                    not candidate["qualified"]
                    or candidate["created"] <= child.created
                    or candidate["coordinates"] != child.coordinates
                    or candidate["hardware_binding"] != child.hardware
                    or candidate["boundary"] != child.boundary
                ):
                    continue
                delta = transfer.scale * (candidate["seconds"] - baseline["seconds"])
                if transfer.contribution + delta < 0:
                    continue
                candidates[candidate["implementation"]].append((transfer, candidate, delta))
            relation["constraints"].append(
                {
                    "kind": "serial-accounting",
                    "observation": parent.identity,
                    "known_component": transfer.child,
                    "known_seconds": transfer.contribution,
                    "remaining_seconds": parent_point["seconds"] - transfer.contribution,
                    "assumptions": transfer.assumptions,
                    "evidence": transfer.evidence,
                }
            )
        scenarios = []
        for implementation, events in sorted(candidates.items()):
            updates = {}
            batches = defaultdict(list)
            for event in events:
                batches[event[1]["created"]].append(event)
            for _, batch in sorted(batches.items()):
                # Every evidence-time update is retained. A later enclosing capture
                # must validate the earlier prediction, never replace its inputs.
                for event in sorted(
                    batch, key=lambda e: e[1].get("observation", str(e[1].get("inputs")))
                ):
                    updates[event[0].child] = event
                scenarios.append((implementation, dict(updates)))
        for implementation, updates in scenarios:
            if parent_point["seconds"] + sum(d for _, _, d in updates.values()) <= 0:
                continue
            dependencies_used = tuple(
                sorted(
                    {
                        ref
                        for _, point, _ in updates.values()
                        for ref in point.get("inputs", (point.get("observation"),))
                        if ref
                    }
                )
            )
            prediction = {
                "baseline": baseline_id,
                "seconds": parent_point["seconds"] + sum(d for _, _, d in updates.values()),
                "rate": parent_point["roofline"]["quantity"]
                / (parent_point["seconds"] + sum(d for _, _, d in updates.values())),
                "roofline": parent_point["roofline"],
                "hardware": parent_point["hardware"],
                "kind": "conditional",
                "qualified": True,
                "implementation": implementation,
                "coordinates": parent.coordinates,
                "hardware_binding": parent.hardware,
                "boundary": parent.boundary,
                "created": max(c["created"] for _, c, _ in updates.values()),
                "inputs": dependencies_used,
                "assumptions": sorted(
                    {
                        "unmeasured siblings and enclosing overhead retain baseline cost",
                        *(a for t, _, _ in updates.values() for a in t.assumptions),
                    }
                ),
                "evidence": sorted({e for t, _, _ in updates.values() for e in t.evidence}),
                "transfers": sorted(t.identity for t, _, _ in updates.values()),
                "validation": [],
            }
            prediction["attainment"] = (
                prediction["rate"] / prediction["roofline"]["ceiling"]
                if prediction["roofline"]["ceiling"]
                else None
            )
            for actual in relation["points"]:
                if (
                    actual["qualified"]
                    and actual["created"] > prediction["created"]
                    and actual["hardware_binding"] == parent.hardware
                    and actual["coordinates"] == parent.coordinates
                    and actual["boundary"] == parent.boundary
                    and actual["implementation"] == implementation
                ):
                    prediction["validation"].append(
                        {
                            "observation": actual["observation"],
                            "error_seconds": actual["seconds"] - prediction["seconds"],
                        }
                    )
            relation["predictions"].append(prediction)
            for source in dependencies_used:
                dependencies[source].add(parent.component)


def observed_transfers(observations, manifests, captures):
    """Connect isolated baselines to same-revision in-parent captured regions.

    This is an explicit conditional transfer: graph geometry and artifact agree,
    while unchanged input-dependent behavior remains an assumption, not a fact.
    """
    from .derive import contributions
    from .records import Transfer, identity

    isolated = [
        o
        for o in observations.values()
        if o.boundary == "isolated-complete-operation"
        and o.correctness == "passed"
        and o.status == "complete"
    ]
    result = {}
    for parent_run in tuple(observations.values()):
        for capture_id in parent_run.captures:
            captured = captures[capture_id]
            manifest = manifests[captured.manifest]
            accounting = contributions(manifest, captured)
            formulas = {f.component: f for f in manifest.formulas}
            native = [
                observations[capture_id + ":" + component]
                for component in formulas
                if capture_id + ":" + component in observations
            ]
            for child in (*native, *isolated):
                if (
                    child.hardware != parent_run.hardware
                    or child.created > parent_run.created
                    or child.implementation != parent_run.implementation
                ):
                    continue
                if manifests[child.manifest].graph != manifest.graph:
                    continue
                if child.coordinates.get("artifact") != parent_run.coordinates.get("artifact"):
                    continue
                if child.coordinates.get("phase") != parent_run.coordinates.get("phase"):
                    continue
                formula = formulas[child.component]
                parent = formula.parent
                if parent is None or parent not in accounting or child.component not in accounting:
                    continue
                own, enclosing = accounting[child.component], accounting[parent]
                if not own["complete_boundary"] or not enclosing["complete_boundary"]:
                    continue
                child_regions = set(own["regions"])
                left = [r for r in captured.regions if r.identity in child_regions]
                right = [
                    r
                    for r in captured.regions
                    if r.identity in set(enclosing["regions"]) - child_regions
                ]
                if any(a.start < b.end and b.start < a.end for a in left for b in right):
                    continue  # Overlap cannot establish a serial replaceable contribution.
                parent_id = capture_id + ":" + parent
                if parent_id not in observations or not own["inclusive_seconds"]:
                    continue
                key = identity((child.identity, parent_id, capture_id))
                result[key] = Transfer(
                    identity=key,
                    child=child.component,
                    parent=parent,
                    baseline_child=child.identity,
                    baseline_parent=parent_id,
                    contribution=own["inclusive_seconds"],
                    relationship="conditional-serial",
                    assumptions=(
                        "same numerical graph, artifact and implementation baseline",
                        "input-dependent behavior and isolated-to-enclosing timing difference "
                        "remain unchanged",
                        "measured non-overlapping region remains serial "
                        "and retains its fusion boundary",
                    ),
                    evidence=(child.identity, parent_run.identity, capture_id),
                )
    # Alternative baselines from repeated executions are independent scenarios.
    # Choose the newest compatible isolated baseline per parent capture and child.
    latest = {}
    for transfer in result.values():
        key = transfer.baseline_parent, transfer.child
        previous = latest.get(key)
        if (
            previous is None
            or (
                observations[transfer.baseline_child].boundary == "isolated-complete-operation"
                and observations[previous.baseline_child].boundary != "isolated-complete-operation"
            )
            or (
                observations[transfer.baseline_child].boundary
                == observations[previous.baseline_child].boundary
                and observations[transfer.baseline_child].created
                > observations[previous.baseline_child].created
            )
        ):
            latest[key] = transfer
    return {t.identity: t for t in latest.values()}
