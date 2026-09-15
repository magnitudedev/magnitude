"""Paired execution of explicitly prepared implementations on one retained boundary."""

import platform
from datetime import UTC, datetime
from time import perf_counter_ns
from uuid import uuid4

from ..formula import units
from .evidence import ObservedMetric, RunEvidence, Scope
from .ownership import exclusive_measurement
from .preparation import NumericalMismatch
from .records import Measurement, Outcome, UnavailableMetric


def compare_prepared(candidates, *, context, store, protocol, blocks=6):
    """Caller owns preparations. This function never selects or compiles a kernel."""
    if len(candidates) < 2 or not 2 <= blocks <= 100:
        raise ValueError("paired comparison requires at least two candidates and blocks")
    names = tuple(candidates)
    first = candidates[names[0]]
    if any(
        item.device is not first.device
        or item.fixture.identity != first.fixture.identity
        or item.options.precision != first.options.precision
        or item.fixture.isolated.target.semantic_identity
        != first.fixture.isolated.target.semantic_identity
        for item in candidates.values()
    ):
        raise ValueError("candidates must share a device, formula, and exact boundary inputs")
    samples = {name: [] for name in names}
    checked = set()
    failure = None
    errors = {}
    orders = []
    with exclusive_measurement():
        try:
            for name, candidate in candidates.items():
                with candidate.inputs() as invocation:
                    try:
                        candidate.check(invocation, protocol)
                        checked.add(name)
                    except NumericalMismatch as numerical_error:
                        if not protocol.measure_invalid:
                            raise
                        errors[name] = str(numerical_error)
                started = perf_counter_ns()
                warmed = 0
                while (
                    warmed < protocol.warmups
                    or perf_counter_ns() - started < protocol.minimum_warmup_seconds * 1e9
                ):
                    warmed += 1
                    with candidate.inputs() as invocation:
                        candidate.retire(candidate.execute(invocation))
            for block in range(blocks):
                order = names if block % 2 == 0 else names[::-1]
                orders.append(order)
                for name in order:
                    candidate = candidates[name]
                    with candidate.inputs() as invocation:
                        samples[name].append(
                            candidate.sample(invocation, kernel_limit=protocol.kernel_limit)
                        )
        except BaseException as error:
            failure = error
            first.device.drain()
    pair_ids = [str(uuid4()) for _ in range(blocks)]
    comparison_protocol = {
        **protocol.model_dump(mode="json"),
        "samples": blocks,
        "pair_ids": pair_ids,
        "orders": [list(o) for o in orders],
    }
    runs = []
    from .runner import measurement_series

    sample_protocol = protocol.model_copy(update={"samples": blocks})
    for name, candidate in candidates.items():
        series = measurement_series(
            candidate.fixture, candidate.device, candidate.options, sample_protocol
        )
        artifacts, unavailable = {}, ()
        try:
            artifacts = {
                kind: store.put_artifact(source.encode())
                for kind, source in candidate.compiled.evidence().items()
            }
        except Exception as artifact_error:
            unavailable = (
                UnavailableMetric(name="compiled-artifacts", reason=str(artifact_error)),
            )
        measurement = Measurement(
            identity=str(uuid4()),
            created=datetime.now(UTC),
            series=series,
            implementation=candidate.implementation,
            outcome=Outcome.FAILED if name in errors or failure is not None else Outcome.COMPLETE,
            checked=name in checked,
            error=errors.get(name)
            or (f"{type(failure).__name__}: {failure}" if failure is not None else None),
            samples=tuple(samples[name]),
            preparation={"boundary": candidate.fixture.identity, "comparison": True},
            artifacts=artifacts,
            unavailable=unavailable,
        )
        store.publish(measurement)
        target = candidate.fixture.isolated.target
        run = RunEvidence(
            identity=str(uuid4()),
            created=measurement.created,
            context=context.model_copy(
                update={
                    "hardware": first.device.evidence_identity,
                    "host": platform.node(),
                    "implementation": candidate.implementation.fingerprint,
                    "workload": context.workload.model_copy(
                        update={"realization": first.fixture.identity}
                    ),
                }
            ),
            scope=Scope(
                kind="formula",
                formula=target.definition,
                semantics=target.semantic_identity,
                occurrence=target.call.occurrence,
            ),
            protocol=comparison_protocol,
            status="complete" if failure is None else "incomplete",
            correctness="passed"
            if name in checked
            else "failed"
            if name in errors
            else "unchecked",
            measurements=(measurement.identity,),
            metrics=(
                ObservedMetric(
                    name="elapsed",
                    unit=units.second,
                    samples=tuple(s.elapsed_ns / 1e9 for s in samples[name]),
                    boundary="complete-operation",
                    basis="paired complete invocation wall time",
                ),
            )
            if samples[name]
            else (),
            attachments={
                "candidate": name,
                "failure": str(failure) if failure is not None else None,
            },
        )
        store.publish_run(run)
        runs.append(run)
    if failure is not None:
        # A failed implementation may have changed a supposedly read-only input.
        for candidate in candidates.values():
            candidate.close()
        raise failure
    return tuple(runs)
