"""Publish analytical graph facts from the production numerical graph.

This adapter owns no model equations and never infers quantities from kernel names.
"""

import json
from collections import defaultdict
from dataclasses import asdict

from formula_performance.records import (
    Capacity,
    Capture,
    Expression,
    Formula,
    Hardware,
    Manifest,
    Obligation,
    Quantity,
    Region,
    Unit,
    identity,
)

from ..formula import FormulaTree
from .bounds import Bounds
from .semantics import formula_work, node_work
from .traffic import boundary_traffic, boundary_traffic_bound


def component_paths(tree):
    paths, siblings = {}, defaultdict(int)
    for target in tree:
        parent = paths.get(target.call.parent, "")
        definition = target.definition.id
        index = siblings[parent, definition]
        siblings[parent, definition] += 1
        # Root decoder is the model, not another execution-phase wrapper.
        path = "" if target.call.parent is None else f"{parent}/{definition}[{index}]".lstrip("/")
        paths[target.call.occurrence] = path
    return paths


def isolated_values(target, isolated, values):
    if values is None:
        return None
    boundary = tuple(dict.fromkeys(p.value for p in target.call.inputs))
    ordered = (*boundary, *(v for n in target.call.nodes for v in target.graph.node(n).outputs))
    return {i: values[v] for i, v in enumerate(ordered) if v in values}


def primary_quantity(graph, target, values=None):
    metric = target.call.metric
    if metric is None:
        raise ValueError(f"formula {target.definition.id} has no declared primary metric")
    declared = next((q for q in target.call.quantities if q.name == metric), None)
    if declared is not None:
        if type(declared.value) is not int:
            raise ValueError("concrete performance publication requires bound formula dimensions")
        return (
            declared.name,
            declared.value,
            Unit(**asdict(declared.unit)),
            "declared formula quantity",
        )
    if metric == "output-elements":
        amount = sum(graph.value(v).spec.elements for v in {p.value for p in target.call.outputs})
        return metric, amount, Unit(name="element", dimension="count"), "formula output elements"
    if metric == "boundary-bytes":
        isolated = target.isolate()
        local = isolated_values(target, isolated, values)
        return (
            metric,
            boundary_traffic(isolated.graph, local or {}),
            Unit(name="byte", dimension="storage"),
            "formula boundary traffic",
        )
    work = formula_work(graph, target.call, values=values).work
    mapping = {
        "floating-work": (work.floating, "FLOP", "floating-work"),
        "matrix-work": (work.matrix, "FLOP", "floating-work"),
        "integer-work": (work.integer, "integer-op", "integer-work"),
        "special-functions": (work.special, "special-function", "special-function-work"),
        "comparisons": (work.comparisons, "comparison", "comparison-work"),
    }
    if metric not in mapping:
        raise ValueError(
            f"declared primary quantity {metric} is missing from {target.definition.id}"
        )
    amount, name, dimension = mapping[metric]
    if not amount.fixed:
        raise ValueError(
            f"primary metric requires concrete controls: {target.definition.id}/{metric}"
        )
    return metric, amount.lower, Unit(name=name, dimension=dimension), "conventional formula work"


def manifest(graph, *, values=None, compiled=None):
    from ..tensor.graph import _stable

    tree = FormulaTree(graph)
    paths = component_paths(tree)
    parameters, obligations, formulas, unresolved = {}, [], [], {}
    outputs = defaultdict(set)
    for target in tree:
        for port in target.call.outputs:
            outputs[port.value].add(target.call.occurrence)
    for node in graph.nodes:
        try:
            work = node_work(graph, node, values=values)
            missing_work = None
        except ValueError as error:
            if node.operation != "sample":
                raise
            from .semantics import UsefulWork

            work = UsefulWork()
            missing_work = str(error)
        dtype = graph.value(node.inputs[0]).spec.dtype.value if node.inputs else "float32"
        for resource, amount, unit in (
            ("matrix-arithmetic", work.matrix, Unit(name="FLOP", dimension="floating-work")),
            (
                "vector-arithmetic",
                Bounds(
                    max(0, work.floating.lower - work.matrix.upper),
                    max(0, work.floating.upper - work.matrix.lower),
                ),
                Unit(name="FLOP", dimension="floating-work"),
            ),
            ("integer-arithmetic", work.integer, Unit(name="integer-op", dimension="integer-work")),
            (
                "special-functions",
                work.special,
                Unit(name="special-function", dimension="special-function-work"),
            ),
            ("comparisons", work.comparisons, Unit(name="comparison", dimension="comparison-work")),
        ):
            if not amount.lower and missing_work is None:
                continue
            key = f"node:{node.id}:{resource}"
            if missing_work is None:
                parameters[key] = amount.lower
            else:
                unresolved[key] = missing_work
            capacity_dtype = (
                dtype
                if resource == "matrix-arithmetic"
                else "uint32"
                if resource == "integer-arithmetic"
                else "float32"
            )
            obligations.append(
                Obligation(
                    identity=key,
                    origins=(str(node.id),),
                    amount=Expression.parameter(key),
                    unit=unit,
                    resource=resource,
                    mappings=(resource + ":" + capacity_dtype,),
                    rule="primitive-conventional-work-v1:" + node.operation,
                    kind="conditional",
                    conditions=(
                        "classical primitive realization with the declared "
                        "conventional arithmetic mapping",
                    ),
                )
            )
    for target in tree:
        call = target.call
        name, amount, unit, meaning = primary_quantity(graph, target, values)
        key = paths[call.occurrence]
        parameter = f"quantity:{key}:{name}"
        parameters[parameter] = amount
        dependencies = set()
        for port in call.inputs:
            dependencies.update(outputs.get(port.value, ()))
        parent = target.parent
        ancestors = set()
        while parent is not None:
            ancestors.add(parent.call.occurrence)
            parent = parent.parent
        dependencies = {d for d in dependencies if d < call.occurrence and d not in ancestors}
        label = target.definition.id.rsplit(".", 1)[-1].replace("_", " ").capitalize()
        if target.definition.id.rsplit(".", 1)[-1] == "block":
            label += " " + key.rsplit("[", 1)[-1].rstrip("]")
        isolated = target.isolate()
        traffic_key = "boundary:" + key
        traffic, missing_traffic = boundary_traffic_bound(
            isolated.graph, isolated_values(target, isolated, values) or {}
        )
        parameters[traffic_key] = traffic
        if missing_traffic:
            unknown_key = traffic_key + ":indexed"
            unresolved[unknown_key] = "; ".join(missing_traffic)
            obligations.append(
                Obligation(
                    identity=unknown_key,
                    origins=tuple(str(n) for n in call.nodes),
                    amount=Expression.parameter(unknown_key),
                    unit=Unit(name="byte", dimension="storage"),
                    resource="memory",
                    mappings=("memory-bandwidth",),
                    boundary=key,
                    rule="indexed-boundary-traffic-v1",
                    kind="conditional",
                    conditions=(
                        "additional unique regions after overlap with the known boundary traffic",
                    ),
                )
            )
        if call.nodes:
            obligations.append(
                Obligation(
                    identity=traffic_key,
                    origins=tuple(str(n) for n in call.nodes),
                    amount=Expression.parameter(traffic_key),
                    unit=Unit(name="byte", dimension="storage"),
                    resource="memory",
                    mappings=("memory-bandwidth",),
                    boundary=key,
                    rule="unique-boundary-traffic-v1",
                    kind="conditional",
                    conditions=(
                        "boundary data crosses the modeled memory path once; ideal internal reuse",
                    ),
                )
            )
        formulas.append(
            Formula(
                component=key,
                occurrence=call.occurrence,
                parent=paths.get(call.parent),
                definition=target.definition.id,
                version=target.definition.version,
                semantics=target.semantic_identity,
                primary=Quantity(
                    name=name,
                    unit=unit,
                    expression=Expression.parameter(parameter),
                    meaning=meaning,
                ),
                nodes=tuple(str(n) for n in call.nodes),
                label=label,
                inputs=tuple(str(p.value) for p in call.inputs),
                outputs=tuple(str(p.value) for p in call.outputs),
                dependencies=tuple(paths[d] for d in sorted(dependencies)),
            )
        )
    return Manifest(
        graph=graph.fingerprint,
        formulas=tuple(formulas),
        obligations=tuple(obligations),
        parameters=parameters,
        unresolved=unresolved,
        numerical_graph={
            **json.loads(json.dumps(_stable(graph))),
            **({"execution": execution_facts(compiled)} if compiled else {}),
        },
    )


def hardware(device, *, capacities=None):
    if capacities is None:
        capacities = device.capacity_evidence
    rates = {}
    profile = device.characterization
    if profile is not None:
        for rate in profile.rates:
            parameter = rate.resource.value + ":" + rate.dtype.value
            if rate.source is not None:
                parameter += ":" + rate.source.fingerprint
            if parameter not in rates or rates[parameter].value < rate.value:
                rates[parameter] = rate
    records = {c.parameter: c for c in capacities}
    for parameter, rate in rates.items():
        records.setdefault(
            parameter,
            Capacity(
                parameter=parameter,
                pool=rate.resource.value,
                unit=Unit(**asdict(rate.unit)),
                value=rate.value,
                kind="achieved",
                provenance=rate.measurement,
                conditions=rate.conditions,
            ),
        )
    return Hardware(
        identity=device.evidence_identity,
        label=device.evidence_identity,
        facts={"compiler": device.compiler_identity},
        capacities=tuple(records.values()),
    )


def capture(manifest, observation, *, capture_id, execution_graph=None, component=""):
    kernels = observation.kernels
    if kernels is None or kernels.attribution != "compiled-order-and-symbols":
        return None
    # Native occurrence numbers belong to this exact production graph.
    components = {f.occurrence: f.component for f in manifest.formulas}
    regions = []
    for i, activity in enumerate(kernels.activities):
        if activity.graph != (execution_graph or manifest.graph) or activity.started_ns is None:
            continue
        if activity.owner not in components:
            continue
        touched = {o for o in activity.origins if o in components}
        parent_occurrences = {f.parent for f in manifest.formulas if f.occurrence in touched}
        leaves = tuple(components[o] for o in touched if components[o] not in parent_occurrences)
        owners = tuple(dict.fromkeys((components[activity.owner], *leaves)))
        regions.append(
            Region(
                identity=f"{capture_id}:{activity.invocation}:"
                f"{activity.dispatch if activity.dispatch is not None else i}",
                owners=owners,
                start=activity.started_ns / 1e9,
                end=activity.ended_ns / 1e9,
            )
        )
    return Capture(
        identity=capture_id,
        component=component,
        manifest=identity(manifest),
        clock=kernels.clock,
        execution_graph=execution_graph or manifest.graph,
        regions=tuple(regions),
        coverage="complete" if len(regions) == len(kernels.activities) else "partial",
    )


def measured_publication(target, boundary, device, measurement, *, compiled=None):
    """Preserve the enclosing numerical graph with a checked isolated observation."""
    from formula_performance.records import Observation, Publication

    ordered = (
        *dict.fromkeys(p.value for p in target.call.inputs),
        *(v for n in target.call.nodes for v in target.graph.node(n).outputs),
    )
    available = (
        boundary._reference.values
        if boundary._reference is not None
        else {key: tensor.reference for key, tensor in boundary.inputs.items()}
    )
    values = {
        original: available[local] for local, original in enumerate(ordered) if local in available
    }
    graph = manifest(target.graph, values=values, compiled=compiled)
    system = hardware(device)
    component = component_paths(FormulaTree(target.graph))[target.call.occurrence]
    captures = tuple(
        c
        for i, sample in enumerate(measurement.samples)
        if (
            c := capture(
                graph,
                sample,
                capture_id=measurement.identity + ":native:" + str(i),
                execution_graph=compiled.graph.fingerprint if compiled else graph.graph,
                component=component,
            )
        )
        is not None
    )
    observation = Observation(
        identity=measurement.identity,
        manifest=identity(graph),
        component=component,
        hardware=identity(system),
        implementation=identity(measurement.implementation),
        created=measurement.created.isoformat(),
        coordinates={"boundary": boundary.identity},
        samples=tuple(s.elapsed_ns / 1e9 for s in measurement.samples),
        boundary="isolated-complete-operation",
        correctness="passed"
        if measurement.checked
        else "failed"
        if (measurement.error or "").startswith("NumericalMismatch:")
        else "unchecked",
        status="complete" if measurement.outcome.value == "complete" else "failed",
        evidence=(measurement.identity,),
        captures=tuple(c.identity for c in captures),
    )
    return Publication(
        manifests=(graph,), hardware=(system,), observations=(observation,), captures=captures
    )


def execution_facts(compiled):
    """Serialize physical provenance without live handles, source I/O or executable code."""
    from ..tensor.graph import _stable

    actions = []
    if compiled.execution_graph is not None:
        for action in compiled.execution_graph.actions:
            actions.append(
                {
                    "index": action.index,
                    "kind": action.kind.value,
                    "predecessors": list(action.predecessors),
                    "unit": action.unit,
                    "value": action.value,
                    "dynamic": action.source_loop is not None,
                }
            )
    units = []
    for index, item in enumerate(compiled._units):
        for call in item.unit.calls:
            operation = call.operation
            units.append(
                {
                    "unit": index,
                    "nodes": list(operation.nodes),
                    "kernel_count": operation.kernel_count,
                    "symbol": operation.definition.name if operation.definition else None,
                    "dynamic": item.source_loop is not None,
                }
            )
    return {
        "graph": json.loads(json.dumps(_stable(compiled.graph))),
        "formula_graph": compiled.formulas.graph.fingerprint,
        "actions": actions,
        "operations": units,
    }
