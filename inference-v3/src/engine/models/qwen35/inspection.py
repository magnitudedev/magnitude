"""Explicit observation and capture of production Qwen forward boundaries."""

from __future__ import annotations

import platform
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from ops.compiler.compilation import CompiledFunction
from ops.lab.fixtures import Fixture, decode_dense


@dataclass(frozen=True)
class Invocation:
    compiled: CompiledFunction
    inputs: tuple
    resources: dict
    weights: dict
    mode: Literal["prefill", "decode"]
    positions: tuple[int, ...]
    lengths: tuple[int, ...]
    physical_rows: int

    def fixture(self, decode_weight) -> Fixture:
        """Snapshot before submit; weight oracle decoding remains independently supplied.

        The callback receives an original graph Value for immutable weights only.
        No captured output is used as the selected formula's correctness oracle.
        """
        graph = self.compiled.formulas.graph
        device = self.compiled.device
        values = {}
        physical = {}
        for identity, resource in zip(graph.inputs, self.inputs, strict=True):
            values[identity] = decode_dense(device.read(resource), resource.spec).copy()
        for identity in graph.resources:
            value = graph.value(identity)
            resource = self.resources[value.name]
            content = device.read(resource)
            values[identity] = decode_dense(content, resource.spec).copy()
            physical[identity] = content
        bindings = {
            identity: self.weights[graph.value(identity).name] for identity in graph.constants
        }
        return Fixture.from_inputs(graph, values, bindings={**bindings, **physical}, capture=decode_weight)


@contextmanager
def inspect_forwards(runtime, *, captured=None, observed=None, kernel_limit=2048):
    """Agent-owned session; collection follows existing forward completion boundaries.

    `captured(invocation)` is an explicit pre-submit inspection hook. Call fixture()
    there to retain starting values. `observed(invocation, observation)` receives
    production timing at normal forward retirement. Capturing tensors and observing
    timings are separate runs so copy/inspection costs cannot pollute a baseline.
    """
    if (captured is None) == (observed is None):
        raise ValueError("choose either boundary capture or production observation")
    if runtime._inspection is not None or runtime._forwards:
        raise ValueError("inspection requires no outstanding forward or existing inspection")
    runtime._inspection = captured, observed, kernel_limit
    try:
        yield
    finally:
        runtime._inspection = None
        if runtime._forwards:
            raise RuntimeError("retire inspected forwards before closing inspection")


def publish_forward(store, context, invocation: Invocation, observation):
    """Publish a production forward to the same store as isolated measurements."""
    from dataclasses import asdict
    from datetime import UTC, datetime
    from uuid import uuid4

    from ops.formula import units
    from ops.lab.archive import RecordedConfiguration
    from ops.lab.evidence import ObservedMetric, RunEvidence, Scope, fingerprint

    tree = invocation.compiled.formulas
    configuration = RecordedConfiguration.capture(
        tree, label=invocation.mode, device=invocation.compiled.device.evidence_identity
    )
    store.publish_configuration(configuration)
    metrics = [
        ObservedMetric(
            name="forward-time",
            unit=units.second,
            samples=(observation.elapsed_ns / 1e9,),
            boundary="prepare-through-forward-retirement",
            basis="production packing, submission, completion and cleanup",
        )
    ]
    if observation.kernels is not None and observation.kernels.busy_ns is not None:
        metrics.append(
            ObservedMetric(
                name="device-busy-time",
                unit=units.second,
                samples=(observation.kernels.busy_ns / 1e9,),
                boundary="native-interval-union",
                basis="union within one native clock",
            )
        )
    run = RunEvidence(
        identity=str(uuid4()),
        created=datetime.now(UTC),
        context=context.model_copy(
            update={
                "hardware": invocation.compiled.device.evidence_identity,
                "host": platform.node(),
                "implementation": fingerprint(
                    {
                        "graph": invocation.compiled.graph.fingerprint,
                        "compiler": invocation.compiled.device.compiler_identity,
                        "dependencies": [asdict(d) for d in invocation.compiled.code_dependencies],
                    }
                ),
                "conditions": {
                    **context.conditions,
                    "positions": list(invocation.positions),
                    "lengths": list(invocation.lengths),
                    "physical_rows": invocation.physical_rows,
                },
            }
        ),
        scope=Scope(kind=invocation.mode),
        protocol={
            "boundary": "prepare-through-forward-retirement",
            "capture": "natural-completion-v1",
        },
        status=observation.status.value,
        correctness="unchecked",
        configuration=configuration.identity,
        metrics=tuple(metrics),
        observations=(observation,),
        attachments={"compiled_graph": invocation.compiled.graph.fingerprint},
        unavailable=("Production observation does not perform an independent numerical check",),
    )
    from formula_performance.records import Observation, Publication, identity
    from ops.performance.publication import manifest, hardware, capture

    graph = manifest(tree.graph, compiled=invocation.compiled)
    system = hardware(invocation.compiled.device)
    captured = capture(graph, observation, capture_id=run.identity + ":native",
                       execution_graph=invocation.compiled.graph.fingerprint)
    publication = Publication(manifests=(graph,), hardware=(system,),
        captures=(captured,) if captured else (), observations=(Observation(
            identity=run.identity, manifest=identity(graph), component="",
            hardware=identity(system), implementation=run.context.implementation,
            created=run.created.isoformat(), coordinates=run.context.conditions,
            samples=(observation.elapsed_ns / 1e9,), boundary="prepare-through-forward-retirement",
            correctness=run.correctness, status="complete" if run.status == "complete" else "failed",
            captures=(captured.identity,) if captured else (), evidence=(run.identity,),
        ),))
    artifact = store.put_artifact(publication.model_dump_json().encode())
    run = run.model_copy(update={"attachments": {**run.attachments, "formula-performance": artifact}})
    store.publish_run(run)
    return run
