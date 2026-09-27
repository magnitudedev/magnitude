"""Scoped formula bounds and explicitly labeled empirical resource references.

No instruction estimator, implementation ranking, or invented hardware peak lives
here. Bounds use mathematical ports/effects and cited device observations.
"""

from ..formula import units
from .records import Ceiling, CeilingKind, Direction


def output_storage_floor(graph) -> int:
    """Fresh escaping output backing under the production invocation ABI.

    Output aliases and repeated ports share backing. Inputs and mutated state are
    borrowed, not new output allocations. This is not a workspace-size estimate.
    """
    borrowed = {graph.alias_root(value) for value in (*graph.inputs, *graph.constants, *graph.resources)}
    backing = {}
    for identity in graph.outputs:
        root = graph.alias_root(identity)
        if root not in borrowed:
            backing[root] = max(backing.get(root, 0), graph.value(identity).spec.storage_nbytes)
    return sum(backing.values())


def formula_ceilings(fixture, device, protocol):
    graph = fixture.isolated.graph
    ceilings, unavailable = [], []
    floor = output_storage_floor(graph)
    if floor:
        ceilings.append(Ceiling(
            metric="reserved-increase", unit=units.byte, value=floor, direction=Direction.LOWER,
            kind=CeilingKind.THEORETICAL, resource="escaping-output-backing",
            revision="formula-output-storage-v1:" + fixture.isolated.target.semantic_identity,
            assumptions=("production invocation ABI allocates fresh escaping output backing",
                         "declared alias views share storage; inputs and state are borrowed",
                         "excludes workspace, allocator rounding, source staging and transfer copies"),
            provenance="derived from this formula's output specifications and primitive alias effects",
        ))
    # The resource model now covers composed and packed formulas, with its
    # demand/rate evidence retained on Measurement.roofline. Storage remains an
    # independent ABI bound rather than a second arithmetic model here.
    return tuple(ceilings), tuple(unavailable)
