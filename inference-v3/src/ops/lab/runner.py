"""One bounded check/measure/persist loop for API and interactive clients."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event
from time import perf_counter_ns
from uuid import uuid4

from ..binding import Binding, Residency
from ..compiler.compilation import CompileOptions
from ..formula import FormulaHandle, FormulaTree, units
from ..performance.semantics import formula_work
from ..runtime.resources import DeviceRuntime
from ..tensor.graph import _stable
from .fixtures import Fixture, FormulaFixture
from .ownership import exclusive_measurement
from .preparation import PreparedFormula
from .records import (
    CacheCondition, Measurement, MeasurementProtocol, Outcome, Phase, Series,
    SourceCondition, UnavailableMetric, UsefulQuantity,
)
from .store import ObservationStore
from .timing import PhaseClock
from .refresh import OperationSources


class MeasurementCancelled(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RunResult:
    measurement: Measurement
    publish_ns: int
    elapsed_ns: int


class MeasurementRunner:
    """Device-thread-owned execution. The outer worker owns code refresh and queueing."""

    def __init__(
        self, fixture: Fixture, device: DeviceRuntime, store: ObservationStore,
        options: CompileOptions, *, protocol: MeasurementProtocol = MeasurementProtocol(),
        prepared_limit: int = 8, reference_bytes: int = 256 << 20,
        _resource_probe: bool = False,
    ):
        if prepared_limit < 1 or reference_bytes < 1:
            raise ValueError("prepared formula cache must have a positive bound")
        device.check()
        self.fixture, self.device, self.store = fixture, device, store
        self.options, self.protocol = options, protocol
        self.prepared_limit = prepared_limit
        self.reference_bytes = reference_bytes
        self._boundaries: OrderedDict[FormulaHandle, FormulaFixture] = OrderedDict()
        self._prepared: OrderedDict[FormulaHandle, PreparedFormula] = OrderedDict()
        # Evidence dependencies outlive native preparation cache entries. Evicting
        # an executable must not make its displayed measurement immune to edits.
        self._dependencies = {}
        self._closed = False
        self._sources = OperationSources()
        self._resource_probe = _resource_probe

    def refresh(self) -> tuple[FormulaHandle, ...]:
        """Install changed Python definitions, then retire affected preparations."""
        self.device.check()
        revision = self._sources.refresh()
        changed = tuple(target for target, dependencies in self._dependencies.items()
                        if any((item.module, item.symbol) in revision.changed
                               for item in dependencies))
        affected = self.invalidate(changed) if changed else ()
        if revision.changed:
            self.device.invalidate_imports(revision.changed)
        return affected

    def _evict(self, target):
        prepared = self._prepared.get(target)
        if prepared is not None:
            prepared.close()
            del self._prepared[target]
        self._boundaries.pop(target, None)

    def _retain(self, target, boundary):
        self._boundaries[target] = boundary
        self._boundaries.move_to_end(target)
        while len(self._boundaries) > 1:
            backing = {}
            for item in self._boundaries.values():
                backing.update(item.storage())
            if len(self._boundaries) <= self.prepared_limit and sum(backing.values()) <= self.reference_bytes:
                break
            self._evict(next(iter(self._boundaries)))
        # An oversized boundary can execute, but must not become a permanently
        # oversized cache entry. The active request owns it until completion.
        return sum(boundary.storage().values()) <= self.reference_bytes

    def pending_sources(self) -> tuple[FormulaHandle, ...]:
        if not self._dependencies:
            return ()
        modules = self._sources.pending_modules()
        return tuple(target for target, dependencies in self._dependencies.items()
                     if any(item.module in modules for item in dependencies))

    def invalidate(self, changed: tuple[FormulaHandle, ...]) -> tuple[FormulaHandle, ...]:
        """Called after authored code refresh, never merely after a history query."""
        self.device.check()
        affected = FormulaTree(self.fixture.root).affected(changed)
        for target in affected:
            prepared = self._prepared.get(target)
            if prepared is not None:
                prepared.close()
                del self._prepared[target]
        return affected

    def _series(self, fixture: FormulaFixture) -> Series:
        sources = []
        for tensor in fixture.inputs.values():
            physical = tensor.physical
            if isinstance(physical, Binding) and physical.residency == Residency.STREAMED:
                for info in dict.fromkeys(plane.span.source.info for plane in physical.planes):
                    sources.append(SourceCondition(
                        binding=physical.value_identity, source=info,
                        cache=CacheCondition.SOURCE_CACHE_UNCONTROLLED,
                    ))
        geometry = tuple(_stable(port.spec) for port in (*fixture.isolated.inputs, *fixture.isolated.outputs))
        device = self.device.evidence_identity
        return Series(
            formula=fixture.isolated.target.definition,
            semantics=fixture.isolated.target.semantic_identity, fixture=fixture.identity,
            device=device, geometry=hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
            precision=self.options.precision, sources=tuple(sources), protocol=self.protocol,
        )

    def _quantities(self, fixture: FormulaFixture):
        target = fixture.isolated.target
        graph = fixture.isolated.graph
        isolated, = FormulaTree(graph).roots
        work = formula_work(graph, isolated.call, values=fixture.reference.values)
        quantities = []
        unavailable = []
        from ..performance.traffic import boundary_traffic

        quantities.append(UsefulQuantity(name="boundary-bytes", amount=boundary_traffic(graph, fixture.reference.values),
                                         unit=units.byte, basis="ideal unique formula boundary reads and writes"))
        for quantity in target.call.quantities:
            if type(quantity.value) is not int:
                raise ValueError("measurement requires concrete formula quantities")
            quantities.append(UsefulQuantity(name=quantity.name, amount=quantity.value,
                                             unit=quantity.unit, basis="declared formula quantity"))
        # Primitive rules are the sole source of arithmetic obligations. Do not
        # substitute an implementation's instruction count for useful work.
        for name, amount, unit in (
            ("floating-work", work.work.floating, units.flop),
            ("matrix-work", work.work.matrix, units.flop),
            ("integer-work", work.work.integer, units.integer_op),
            ("special-functions", work.work.special, units.special_op),
            ("comparisons", work.work.comparisons, units.comparison),
        ):
            if amount.fixed and math.isfinite(amount.lower):
                if amount.lower:
                    quantities.append(UsefulQuantity(name=name, amount=amount.lower, unit=unit,
                                                     basis="formula primitive useful-work convention"))
            else:
                unavailable.append(UnavailableMetric(
                    name=name, reason="; ".join(work.work.issues) or "not a fixed mathematical work count",
                ))
        return tuple(quantities), tuple(unavailable)

    def measure(
        self, target: FormulaHandle, *, cancellation: Event | None = None,
        progress: Callable[[Phase, int, int], None] | None = None,
    ) -> RunResult:
        self.device.check()
        if self._closed:
            raise RuntimeError("measurement runner is closed")
        if not isinstance(target, FormulaHandle) or target.graph is not self.fixture.root:
            raise ValueError("measurement requires a typed occurrence in the fixture trace")
        started = perf_counter_ns()
        clock = PhaseClock()

        def checkpoint(phase, current=0, total=1):
            if progress is not None:
                progress(phase, current, total)
            if cancellation is not None and cancellation.is_set():
                raise MeasurementCancelled("measurement cancelled")

        # Fixture metadata establishes a real series before any numerical check or
        # compilation. Errors before this point remain worker preparation failures,
        # not performance observations with invented fixture identities.
        with clock.track(Phase.PREPARE):
            boundary = self._boundaries.get(target)
            if boundary is None:
                boundary = self.fixture.boundary(target)
            series = self._series(boundary)
        samples = []
        checked = False
        prepared = None
        quantities, unavailable = (), ()
        outcome, error = Outcome.COMPLETE, None
        retain = False
        roofline = None
        with exclusive_measurement():
            try:
                checkpoint(Phase.PREPARE)
                with clock.track(Phase.PREPARE):
                    _ = boundary.reference
                    quantities, unavailable = self._quantities(boundary)
                    retain = self._retain(target, boundary)
                if not self._resource_probe:
                    from .roofline import model

                    checkpoint(Phase.CHARACTERIZE)
                    with clock.track(Phase.CHARACTERIZE):
                        if self.device.characterization is None:
                            self.device.characterize(self.store, cancellation=cancellation)
                        spans = tuple(plane.span for tensor in boundary.inputs.values()
                                      if isinstance(tensor.physical, Binding) and
                                      tensor.physical.residency == Residency.STREAMED
                                      for plane in tensor.physical.planes)
                        if spans:
                            self.device.characterize_sources(self.store, spans, cancellation=cancellation)
                        roofline = model(boundary, self.device)
                prepared = self._prepared.get(target)
                if prepared is None:
                    checkpoint(Phase.COMPILE)
                    # Retire an idle cache entry before allocating its replacement.
                    # No active request/model/prefix eviction is performed here.
                    with clock.track(Phase.COMPILE):
                        prepared = PreparedFormula(boundary, self.device, self.options)
                    self._prepared[target] = prepared
                    self._dependencies[target] = prepared.code_dependencies
                self._prepared.move_to_end(target)

                def inputs(stack):
                    with clock.track(Phase.CONDITION):
                        return stack.enter_context(prepared.inputs())

                checkpoint(Phase.CHECK)
                with ExitStack() as retained:
                    invocation = inputs(retained)
                    with clock.track(Phase.CHECK):
                        prepared.check(invocation, self.protocol)
                checked = True

                conditioning_started = perf_counter_ns()
                index = 0
                while (index < self.protocol.warmups or
                       perf_counter_ns() - conditioning_started < self.protocol.minimum_warmup_seconds * 1e9):
                    checkpoint(Phase.CONDITION, min(index, self.protocol.warmups), self.protocol.warmups)
                    with clock.track(Phase.CONDITION), prepared.inputs() as invocation:
                        execution = prepared.execute(invocation)
                        prepared.retire(execution)
                    index += 1
                for index in range(self.protocol.samples):
                    checkpoint(Phase.SAMPLE, index, self.protocol.samples)
                    with ExitStack() as retained:
                        invocation = inputs(retained)
                        with clock.track(Phase.SAMPLE):
                            samples.append(prepared.sample(invocation, kernel_limit=self.protocol.kernel_limit))
                checkpoint(Phase.SAMPLE, self.protocol.samples, self.protocol.samples)
            except MeasurementCancelled as failure:
                outcome, error = Outcome.CANCELLED, str(failure)
                self.device.drain()
            except Exception as failure:
                outcome, error = Outcome.FAILED, f"{type(failure).__name__}: {failure}"
                # A broken wait must not be hidden by dropping pinned resources.
                # If drain also fails, propagate that runtime failure to the worker.
                self.device.drain()

        ceilings = ()
        if outcome != Outcome.COMPLETE and prepared is not None:
            # A failed implementation may have corrupted a supposedly read-only
            # input. Preserve its evidence/reference, never its live preparation.
            prepared.close()
            self._prepared.pop(target, None)
        if outcome == Outcome.COMPLETE:
            from .ceilings import formula_ceilings

            ceilings, missing = formula_ceilings(boundary, self.device, self.protocol)
            unavailable = (*unavailable, *missing)
            if any(sample.kernels is None for sample in samples):
                unavailable = (*unavailable, UnavailableMetric(
                    name="kernel-device-time",
                    reason="disabled by measurement protocol" if self.protocol.kernel_limit is None else
                           "selected native endpoint does not expose compute-pass timestamps",
                ))
        measurement = Measurement(
            identity=str(uuid4()), created=datetime.now(UTC), series=series,
            implementation=prepared.implementation if prepared is not None else None,
            outcome=outcome, checked=checked, samples=tuple(samples), quantities=quantities, ceilings=ceilings,
            unavailable=unavailable, phases=clock.snapshot(), error=error,
            roofline=roofline,
        )
        if not retain:
            self._evict(target)
        if progress is not None:
            progress(Phase.PUBLISH, 0, 1)
        publish_started = perf_counter_ns()
        self.store.publish(measurement)
        finished = perf_counter_ns()
        return RunResult(measurement, finished - publish_started, finished - started)

    def close(self) -> None:
        self.device.check()
        if self._closed:
            return
        self.device.drain()
        for target, prepared in tuple(self._prepared.items()):
            prepared.close()
            del self._prepared[target]
        self._boundaries.clear()
        self._closed = True
