"""One hardware-parameterized bound evaluator and exact execution attribution."""

from collections import defaultdict

from .records import Formula, Hardware, Manifest


def roofline(manifest: Manifest, formula: Formula, hardware: Hardware):
    """Compose unique primitive obligations before binding their hardware capacities.

    Alternative mappings cannot be added as simultaneous demands. Shared-pool terms
    add only when every admitted mapping has the same pool. Cross-pool alternatives
    supply independent lower bounds; this intentionally retains legal overlap.
    """
    nodes = set(formula.nodes)
    capacities = {c.parameter: c for c in hardware.capacities}
    bounds, empirical = defaultdict(float), defaultdict(float)
    terms, missing, assumptions = [], [], set(manifest.conditions)
    conditional = False
    for obligation in manifest.obligations:
        if obligation.boundary is not None and obligation.boundary != formula.component:
            continue
        overlap = set(obligation.origins) & nodes
        if not overlap:
            continue
        if overlap != set(obligation.origins):
            # Joint obligations cannot be attributed to an arbitrary child slice.
            continue
        try:
            amount = obligation.amount.evaluate(manifest.parameters)
        except KeyError as error:
            missing.append(
                {
                    "obligation": obligation.identity,
                    "parameter": error.args[0],
                    "reason": manifest.unresolved[error.args[0]],
                }
            )
            continue
        if not amount:
            continue
        candidates = [capacities.get(name) for name in obligation.mappings]
        if any(c is None for c in candidates):
            missing.append(
                {
                    "obligation": obligation.identity,
                    "capacities": [name for name in obligation.mappings if name not in capacities],
                }
            )
            continue
        if any(
            c.unit.name != obligation.unit.name + "/s"
            or c.unit.dimension != obligation.unit.dimension + "/time"
            for c in candidates
        ):
            raise ValueError("capacity and obligation units disagree")
        exact = [c for c in candidates if c.kind != "achieved"]
        # An achieved alternative does not establish an upper envelope for that mapping.
        is_bound = len(exact) == len(candidates)
        chosen = max(exact if is_bound else candidates, key=lambda c: c.value)
        seconds = amount / chosen.value
        pools = {c.pool for c in candidates}
        key = next(iter(pools)) if len(pools) == 1 else "alternative:" + obligation.identity
        target = bounds if is_bound else empirical
        target[key] += seconds
        conditions = (*obligation.conditions, *(a for c in candidates for a in c.conditions))
        assumptions.update(conditions)
        conditional |= is_bound and (
            obligation.kind == "conditional"
            or any(c.kind == "conditional-upper-bound" for c in candidates)
        )
        terms.append(
            {
                "obligation": obligation.identity,
                "origins": obligation.origins,
                "amount": amount,
                "unit": obligation.unit.model_dump(),
                "capacity_parameters": obligation.mappings,
                "pool": key,
                "seconds": seconds,
                "kind": "bound" if is_bound else "empirical-reference",
                "rule": obligation.rule,
                "conditions": conditions,
                "capacities": [c.model_dump() for c in candidates],
            }
        )
    certificates = []
    for serial in manifest.serial_stages:
        if serial.component != formula.component:
            continue
        stages = []
        for obligations in serial.stages:
            stage_pools = defaultdict(float)
            for term in terms:
                if term["kind"] == "bound" and term["obligation"] in obligations:
                    stage_pools[term["pool"]] += term["seconds"]
            stages.append(max(stage_pools.values(), default=0))
        certificates.append(
            {
                "certificate": serial.identity,
                "seconds": sum(stages),
                "stages": stages,
                "rule": serial.rule,
                "conditions": serial.conditions,
            }
        )
        assumptions.update(serial.conditions)
        conditional |= bool(serial.conditions)
    floor = max((*bounds.values(), *(c["seconds"] for c in certificates)), default=0)
    reference = max(empirical.values(), default=0)
    quantity = formula.primary.expression.evaluate(manifest.parameters)
    return {
        "quantity": quantity,
        "metric": formula.primary.name,
        "unit": formula.primary.unit.model_dump(),
        "floor_seconds": floor or None,
        "ceiling": quantity / floor if floor and quantity else None,
        "kind": "conditional" if conditional and floor else "established" if floor else "unbounded",
        "reference_seconds": reference or None,
        "reference_rate": quantity / reference if reference and quantity else None,
        "terms": terms,
        "serial_certificates": certificates,
        "missing": missing,
        "assumptions": sorted(assumptions),
        "expression": "Q_F(x) / max(resource-pool bounds, certified sequential-stage bounds)",
        "hardware_parameters": {c.parameter: c.value for c in hardware.capacities},
        "limiting_pool": max(bounds, key=bounds.get) if bounds else None,
    }


def interval_union(regions):
    total = end = 0.0
    for region in sorted(regions, key=lambda r: r.start):
        total += max(0, region.end - max(end, region.start))
        end = max(end, region.end)
    return total


def contributions(manifest, capture):
    """A fused region remains shared at its smallest containing formula."""
    formulas = {f.component: f for f in manifest.formulas}

    def ancestors(component):
        result = {component}
        while formulas[component].parent is not None:
            component = formulas[component].parent
            result.add(component)
        return result

    paths = {c: ancestors(c) for c in formulas}
    owned = defaultdict(list)
    for region in capture.regions:
        common = set.intersection(*(paths[o] for o in region.owners))
        if not common:
            raise ValueError("execution region has no containing formula")
        owner = max(common, key=lambda c: len(paths[c]))
        owned[owner].append(region)
    result = {}
    for component, formula in formulas.items():
        if capture.component not in paths[component]:
            continue
        inclusive = [
            r for owner, regions in owned.items() if component in paths[owner] for r in regions
        ]
        parent_regions = (
            [
                r
                for owner, regions in owned.items()
                if formula.parent in paths[owner]
                for r in regions
            ]
            if formula.parent is not None and component != capture.component
            else []
        )
        seconds = interval_union(inclusive)
        parent_seconds = interval_union(parent_regions)
        crossing = [
            r.identity
            for r in capture.regions
            if any(component in paths[o] for o in r.owners)
            and not all(component in paths[o] for o in r.owners)
        ]
        result[component] = {
            "capture": capture.identity,
            "clock": capture.clock,
            "coverage": capture.coverage,
            "inclusive_seconds": seconds,
            "exclusive_seconds": interval_union(owned[component]),
            "parent_seconds": parent_seconds
            if formula.parent is not None and component != capture.component
            else None,
            "parent_fraction": seconds / parent_seconds if parent_seconds else None,
            "regions": [r.identity for r in inclusive],
            "joint_regions": [r.identity for r in owned[component] if len(r.owners) > 1],
            "crossing_regions": crossing,
            "complete_boundary": capture.coverage == "complete" and not crossing,
        }
    return result


def formula_model(manifest, formula):
    """Hardware-independent demand facts, available before any direct measurement.

    Amounts describe unique numerical obligations. Alternative mappings remain
    alternatives; only roofline() binds them to capacity pools and latency bounds.
    """
    nodes = set(formula.nodes)
    demands = {}
    for obligation in manifest.obligations:
        if not set(obligation.origins) <= nodes or (
            obligation.boundary is not None and obligation.boundary != formula.component
        ):
            continue
        key = (
            obligation.resource,
            obligation.unit.name,
            obligation.unit.dimension,
            obligation.mappings,
            obligation.kind,
            obligation.conditions,
        )
        demand = demands.setdefault(
            key,
            {
                "resource": obligation.resource,
                "unit": obligation.unit.model_dump(),
                "capacity_parameters": obligation.mappings,
                "known_amount": 0,
                "unresolved": [],
                "kind": obligation.kind,
                "conditions": obligation.conditions,
                "rules": set(),
            },
        )
        demand["rules"].add(obligation.rule)
        try:
            demand["known_amount"] += obligation.amount.evaluate(manifest.parameters)
        except KeyError as error:
            demand["unresolved"].append(manifest.unresolved[error.args[0]])
    return {
        "quantity": formula.primary.expression.evaluate(manifest.parameters),
        "unit": formula.primary.unit.model_dump(),
        "demands": [
            {**d, "rules": sorted(d["rules"]), "unresolved": sorted(set(d["unresolved"]))}
            for d in demands.values()
        ],
        "sequential_certificates": [
            s.model_dump() for s in manifest.serial_stages if s.component == formula.component
        ],
    }
