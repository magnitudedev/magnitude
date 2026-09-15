"""Publish existing session observations into the formula performance evidence store."""

import platform
from datetime import UTC, datetime
from pathlib import Path

from ops.formula import units
from ops.lab.evidence import (
    ExecutionContext,
    Model,
    ObservedMetric,
    RunEvidence,
    Scope,
    Workload,
    fingerprint,
)
from ops.lab.store import ObservationStore


def publish_session(
    path: Path, model: str, plan, adapter, run_store, request, observation, block: int
):
    """The request and observation are the same objects used by session-bench."""
    artifact = adapter.artifact
    artifact_id = fingerprint([(f.path, f.sha256) for f in artifact.files])
    recipe = {"plan": plan.model_dump(mode="json"), "request": request.id}
    context = ExecutionContext(
        model=Model(identity=model, label=model),
        workload=Workload(
            kind="benchmark",
            recipe=recipe,
            realization=fingerprint(
                {
                    "body": request.body("model"),
                    "artifact": artifact_id,
                    "engine": adapter.target.engine,
                }
            ),
        ),
        engine=adapter.target.engine,
        artifact=artifact_id,
        numerical_contract=fingerprint(artifact.metadata),
        hardware=fingerprint(run_store.hardware),
        host=platform.node(),
        implementation=fingerprint(adapter.identity),
        conditions={
            "hardware": run_store.hardware,
            "artifact": artifact.metadata,
            "checkpoint": request.checkpoint,
            "concurrency": request.concurrency,
            "cache_policy": plan.cache_policy,
        },
    )
    metrics = []
    for name, value in (
        ("first-token-latency", observation.ttft_ms),
        ("completed-latency", observation.completed_ms),
    ):
        if value is not None:
            metrics.append(
                ObservedMetric(
                    name=name,
                    unit=units.second,
                    samples=(value / 1000,),
                    boundary="http-" + name,
                    basis="client-observed elapsed wall time",
                )
            )
    if observation.terminal:
        timing = observation.terminal["timings"]
        for name, key, count in (
            ("prefill-time", "prompt_ms", "prompt_n"),
            ("decode-time", "predicted_ms", "predicted_n"),
        ):
            if key in timing:
                metrics.append(
                    ObservedMetric(
                        name=name,
                        unit=units.second,
                        samples=(timing[key] / 1000,),
                        counts=(timing[count],),
                        boundary=adapter.timing_basis,
                        basis=(
                            f"source timer {key}; source count {count}; "
                            "no interval reinterpretation"
                        ),
                    )
                )
    run = RunEvidence(
        identity=fingerprint((run_store.path.name, adapter.target.id, request.id, block)),
        created=datetime.now(UTC),
        context=context,
        scope=Scope(kind="request"),
        protocol={"boundary": "session-http-v1", "timing_basis": adapter.timing_basis},
        status="complete" if observation.outcome in ("valid", "invalid") else "incomplete",
        correctness="unchecked",  # Serving validation is not an independent numerical oracle.
        metrics=tuple(metrics),
        attachments={
            "session_validation": observation.model_dump(mode="json"),
            "engine_identity": adapter.identity,
            "source_directory": str(run_store.path),
        },
        unavailable=(
            "Formula attribution is unavailable for opaque request observations",
            "Rendered token/state identity is not established by HTTP request equality",
        ),
    )
    with ObservationStore(path) as store:
        store.publish_run(run)
