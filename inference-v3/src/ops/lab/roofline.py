"""Formula demands joined with measured resource references, independent of kernels."""

import math

from ..binding import Binding, Residency
from ..formula import Unit, units
from ..performance.bounds import Bounds
from ..performance.resources import Resource
from ..performance.semantics import node_work
from ..performance.traffic import boundary_accesses, boundary_traffic, unique_bytes
from ..tensor.types import DType
from .records import ResourceDemand, ResourceLimit, Roofline


class RooflineCoverageError(ValueError):
    """Missing model inputs cannot become a successful efficiency report."""


def demands(fixture):
    graph = fixture.isolated.graph
    values = fixture.reference.values
    totals = {}

    def add(resource, dtype, amount, unit):
        if not math.isfinite(amount.upper):
            raise RooflineCoverageError(f"{resource.value} requires finite concrete work")
        if amount.upper:
            key = resource, dtype, unit
            totals[key] = totals.get(key, Bounds.exact(0)) + amount

    for node in graph.nodes:
        work = node_work(graph, node, values=values)
        dtype = graph.value(node.inputs[0]).spec.dtype if node.inputs else DType.F32
        if not work.matrix.fixed or not work.floating.fixed:
            raise RooflineCoverageError(f"{node.operation} arithmetic requires concrete work counts")
        add(Resource.MATRIX_ARITHMETIC, dtype, work.matrix, units.flop)
        add(Resource.VECTOR_ARITHMETIC, DType.F32,
            Bounds.exact(work.floating.lower - work.matrix.lower), units.flop)
        add(Resource.INTEGER_ARITHMETIC, DType.U32, work.integer, units.integer_op)
        add(Resource.SPECIAL_FUNCTIONS, DType.F32, work.special, units.special_op)
        add(Resource.COMPARISONS, DType.F32, work.comparisons, units.comparison)
    traffic = boundary_traffic(graph, values)
    add(Resource.EXECUTION_COPY, DType.U8, Bounds.exact(traffic), units.byte)
    result = tuple(ResourceDemand(resource=resource, dtype=dtype, lower=amount.lower, upper=amount.upper,
                                unit=unit, basis=("ideal unique boundary traffic; internal intermediates retained on chip"
                                                 if resource == Resource.EXECUTION_COPY else
                                                 "formula primitive conventional useful work; not emitted instructions"))
                 for (resource, dtype, unit), amount in totals.items())
    reads, _ = boundary_accesses(graph, values)
    sources = {}
    for identity, tensor in fixture.inputs.items():
        binding = tensor.physical
        if not isinstance(binding, Binding) or binding.residency != Residency.STREAMED:
            continue
        density = tensor.spec.storage_nbytes / tensor.spec.elements
        for plane in binding.planes:
            regions = sources.setdefault(plane.span.source.info, [])
            for first, last in reads.get(graph.alias_root(identity), ()):
                begin = math.floor(first / density / plane.group_elements) * plane.group_bytes
                end = math.ceil(last / density / plane.group_elements) * plane.group_bytes
                if end > plane.span.length:
                    raise RooflineCoverageError("source groups exceed the bound plane")
                regions.append((plane.span.offset + begin, plane.span.offset + end))
    return (*result, *(ResourceDemand(resource=Resource.SOURCE_IMPORT, dtype=DType.U8,
                                     lower=unique_bytes(regions), upper=unique_bytes(regions), unit=units.byte,
                                     source=info, basis="unique encoding-aligned source groups required by this formula")
                       for info, regions in sources.items() if unique_bytes(regions)))


def model(fixture, device):
    profile = device.characterization
    if (profile is None or profile.device != device.evidence_identity or
            profile.compiler != device.compiler_identity or
            profile.compiler_target != device.compiler_target.identity):
        raise RooflineCoverageError("load/measure resource characterization for this device and compiler first")
    limits = []
    for demand in demands(fixture):
        expected_unit = Unit(f"{demand.unit.name}/s", f"{demand.unit.dimension}/time")
        matching = tuple(rate for rate in profile.rates if rate.resource == demand.resource and
                         (rate.dtype == demand.dtype or demand.resource in {Resource.EXECUTION_COPY, Resource.SOURCE_IMPORT})
                         and rate.unit == expected_unit and rate.source == demand.source)
        widened = False
        if not matching and demand.resource == Resource.MATRIX_ARITHMETIC and demand.dtype == DType.BF16:
            # Exact BF16 -> FP32 widening is a legal arithmetic realization when
            # this device has no native BF16 matrix instruction. Never substitute
            # an FP16 peak, which does not preserve BF16's exponent range.
            matching = tuple(rate for rate in profile.rates if rate.resource == demand.resource and
                             rate.dtype == DType.F32 and rate.unit == expected_unit)
            widened = True
        if not matching:
            raise RooflineCoverageError(f"missing {demand.resource.value}/{demand.dtype.value} resource evidence")
        # A resource ceiling uses the best observed rate, not an execution-time
        # prediction from a similarly sized (possibly launch-bound) probe.
        rate = max(matching, key=lambda rate: rate.value)
        assumptions = (*rate.conditions, *(('BF16 matrix demand uses exact FP32 widening reference',) if widened else ()))
        limits.append(ResourceLimit(demand=demand, rate=rate.value, rate_dtype=rate.dtype,
                                    measurement=rate.measurement, assumptions=assumptions))
    return Roofline(revision="formula-resource-roofline-v2", characterization=profile.identity,
                    limits=tuple(limits), assumptions=(
                        "conventional formula work divided by empirical resource references, not an absolute physical proof",
                        "shared precision variants add within each resource; maximum across resource constraints assumes ideal overlap",
                        "boundary traffic assumes ideal reuse and on-chip intermediates; cache residency is uncontrolled",
                        "packed decode, extra passes, launches and allocation overhead remain implementation costs",
                        "vector/special work uses FP32 and integer work uses the uint32 conventional reference",
                        "isolated children are independently compiled; their times are not contributions to their parent",
                        "ratios above 100% challenge calibration/model applicability and are never clamped",
                    ))
