"""One managed run owns preparation, sequential engines, and concurrent simulated sessions."""

import asyncio
import contextlib
import fcntl
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from benchmark_fixtures import bfcl as corpus
from benchmark_fixtures import prose as prose_source
from benchmark_fixtures.prose_history import Prose
from benchmark_fixtures.ruler import RetrievalAnswers, RulerFixture
from performance.thermals import ThermalRecorder

from . import report
from .client import Observation, measure
from .engines import ADAPTERS
from .engines.base import command as subprocess_command
from .engines.base import runtime_digest
from .models import Artifact, Target
from .policy import (
    CONTEXT_ALIGNMENT,
    MAX_OUTPUT_TOKENS,
    PROSE_OUTPUT_TOKENS,
    RETRIEVAL_OUTPUT_TOKENS,
)
from .results import RunStore, atomic_json, public_command
from .sessions import Plan, Request
from .suites import compile_plan


@contextlib.contextmanager
def machine_lock():
    # This lock is process coordination only, never alias configuration or result storage.
    path = Path(tempfile.gettempdir()) / f"magnitude-inference-measurement-{os.getuid()}.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another session-bench run owns this machine") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def capacity(
    counts: list[dict[str, int]], limits: list[int], output_limit: int = MAX_OUTPUT_TOKENS
) -> int:
    if not counts or any(
        not values or any(type(n) is not int or n < 1 for n in values.values()) for values in counts
    ):
        raise ValueError("adapter did not provide valid rendered prompt counts")
    required = max(max(values.values()) for values in counts) + output_limit
    required = (required + CONTEXT_ALIGNMENT - 1) // CONTEXT_ALIGNMENT * CONTEXT_ALIGNMENT
    if required > min(limits):
        raise ValueError(
            f"requests need {required} context tokens including {output_limit} "
            f"output headroom; smallest model limit is {min(limits)}"
        )
    return required


async def execute(
    plan: Plan,
    adapter,
    engine,
    store: RunStore,
    block: int,
    records: list[dict],
    progress: Callable[[str], None],
) -> None:
    settled = {}
    tasks = {}
    recorded = set()
    started = time.perf_counter()

    def save(request: Request, observation: Observation, phase="measured"):
        if (phase, request.id) in recorded:
            return
        recorded.add((phase, request.id))
        if observation.outcome == "cancelled" and engine.process.returncode is not None:
            observation = observation.model_copy(
                update={
                    "outcome": "target-failure",
                    "error": "engine exited during execution",
                }
            )
        row = {
            "target": adapter.target.id,
            "block": block,
            "phase": phase,
            "section": request.section,
            "workload": request.workload,
            "checkpoint": request.checkpoint,
            "concurrency": request.concurrency,
            "session": request.session,
            "fixture_id": request.fixture_id,
            "timing_basis": adapter.timing_basis,
            "observation": observation.model_dump(mode="json", exclude_none=True),
        }
        if isinstance(request.expected, RetrievalAnswers):
            row["retrieval_total"] = len(request.expected.values)
        records.append(row)
        store.append("results.jsonl", row)
        store.event(
            "request_finished",
            target=adapter.target.id,
            request=request.id,
            block=block,
            phase=phase,
            outcome=observation.outcome,
        )
        detail = ""
        if observation.retrieval is not None:
            score = observation.retrieval
            detail = f"; retrieval {score.correct}/{score.total}, exact={score.exact_match}"
        progress(f"{adapter.target.id} {request.id}: {observation.outcome}{detail}")

    async with httpx.AsyncClient(
        timeout=None,
        trust_env=False,
        limits=httpx.Limits(max_connections=plan.parallel_sequences + 1),
    ) as client:
        warmup = plan.warmup
        obs = await measure(
            client,
            engine.endpoint,
            engine.model,
            warmup,
            lambda event: store.record_stream(
                "logs/warmup-streams.jsonl", {"target": adapter.target.id, "block": block, **event}
            ),
            extensions=adapter.extensions,
            cancelled=lambda obs: save(warmup, obs, "warmup"),
        )
        save(warmup, obs, "warmup")
        if obs.outcome != "valid":
            raise RuntimeError(f"engine qualification request failed: {obs.outcome}: {obs.error}")

        async def run_request(request):
            for dependency in request.depends_on:
                await tasks[dependency]
                if settled[dependency] not in ("valid", "invalid"):
                    obs = Observation(
                        request_id=request.id,
                        outcome="dependency-failed",
                        error=f"dependency failed: {dependency}",
                    )
                    save(request, obs)
                    settled[request.id] = obs.outcome
                    return
            delay = request.release_ms / 1000 - (time.perf_counter() - started)
            if delay > 0:
                await asyncio.sleep(delay)
            store.event(
                "request_started", target=adapter.target.id, block=block, request=request.id
            )
            stream_path = f"logs/{adapter.target.id}-b{block}-{request.id}.jsonl"
            obs = await measure(
                client,
                engine.endpoint,
                engine.model,
                request,
                lambda event: store.record_stream(stream_path, event),
                extensions=adapter.extensions,
                cancelled=lambda obs: save(request, obs),
            )
            if (
                obs.terminal
                and obs.terminal["usage"]["prompt_tokens_details"]["cached_tokens"] != 0
            ):
                obs = obs.model_copy(
                    update={
                        "outcome": "protocol-error",
                        "error": "engine reported cached tokens under the cache-disabled policy",
                    }
                )
            save(request, obs)
            settled[request.id] = obs.outcome

        started = time.perf_counter()
        try:
            # Sections are independent schedules and cannot overlap accidentally.
            for section in dict.fromkeys(request.section for request in plan.requests):
                async with asyncio.TaskGroup() as group:
                    for request in plan.requests:
                        if request.section == section:
                            tasks[request.id] = group.create_task(run_request(request))
        finally:
            # Include requests that never reached submission when the target/run was interrupted.
            for request in plan.requests:
                if ("measured", request.id) not in recorded:
                    save(
                        request,
                        Observation(
                            request_id=request.id,
                            outcome="cancelled",
                            error="execution interrupted before completion",
                        ),
                    )


async def run(
    root: Path,
    targets: list[Target],
    sections: tuple[str, ...],
    contexts: tuple[int, ...],
    categories: tuple[str, ...],
    repeat: int,
    case: str | None,
    progress: Callable[[str], None],
    prose: bool = False,
    retrieval: RulerFixture | None = None,
    needle_depth: float = 0.5,
) -> dict:
    if retrieval is not None and (prose or categories or case is not None):
        raise ValueError("retrieval cannot be combined with prose or tool selection")
    command = public_command(
        targets, sections, contexts, categories, repeat, case, prose, retrieval, needle_depth
    )
    workload = "retrieval" if retrieval else "prose" if prose else "tools"
    store = RunStore(
        root,
        command,
        {
            "targets": [t.model_dump() for t in targets],
            "sections": sections,
            "contexts": contexts,
            "categories": categories,
            "repeat": repeat,
            "case": case,
            "workload": workload,
            "retrieval": retrieval.model_dump(mode="json") if retrieval else None,
            "needle_depth": needle_depth if retrieval else None,
            "max_output_tokens": (
                RETRIEVAL_OUTPUT_TOKENS
                if retrieval
                else PROSE_OUTPUT_TOKENS
                if prose
                else MAX_OUTPUT_TOKENS
            ),
        },
    )
    progress(f"Run: {store.path}")
    records = []
    planned = 0
    status, error = "failed", None
    thermals = ThermalRecorder(store.path)
    try:
        with machine_lock(), thermals:
            store.snapshot("session-bench", root)
            source_identity = runtime_digest(root)
            if retrieval is not None:
                progress("Preparing RULER-derived retrieval fixtures")
                fixtures = retrieval
                corpus_digest = retrieval.identity
            elif prose:
                if case is not None:
                    raise ValueError("--case is only supported for tool fixtures")
                progress("Preparing pinned Moby Dick")
                text, provenance = await prose_source.prepare()
                fixtures = Prose(text, provenance)
                corpus_digest = fixtures.identity
            else:
                progress("Preparing pinned BFCL corpus")
                fixtures, corpus_digest = await corpus.prepare(categories)
            adapters = {}
            counts = []
            limits = []
            for target in targets:
                progress(f"Preparing {target.engine}: {target.reference}")
                artifact_record = store.path / f"{target.id}-artifact.json"
                await subprocess_command(
                    [
                        sys.executable,
                        "-m",
                        "session_bench.models",
                        target.model_dump_json(),
                        str(artifact_record),
                    ],
                    root,
                    store.path / "logs" / f"{target.id}-artifact.log",
                )
                artifact = Artifact.model_validate_json(artifact_record.read_text())
                adapter = ADAPTERS[target.engine](root, target, artifact, store)
                await adapter.prepare()
                limits.append(artifact.context_limit)
                atomic_json(store.path / f"{target.id}-runtime.json", adapter.identity)
                adapters[target.id] = adapter
            sizing = adapters[targets[0].id]
            async with sizing.context_counter() as counter:
                plan = await compile_plan(
                    fixtures,
                    corpus_digest,
                    sections,
                    contexts,
                    case,
                    counter=counter,
                    sizing_identity=sizing.target.id,
                    needle_depth=needle_depth,
                )
            for request in plan.prepared_requests:
                store.append("requests.jsonl", request.model_dump(mode="json"))
            blocks = max(2, len(targets)) * repeat
            order = [
                targets[i % len(targets) :] + targets[: i % len(targets)] for i in range(blocks)
            ]
            planned = len(plan.requests) * len(targets) * blocks
            atomic_json(
                store.path / "plan.json",
                {
                    "digest": plan.identity,
                    "parallel_sequences": plan.parallel_sequences,
                    "cache_policy": plan.cache_policy,
                    "corpus_digest": corpus_digest,
                    "planned_requests": planned,
                    "execution_order": [[target.id for target in row] for row in order],
                },
            )
            for target in targets:
                adapter = adapters[target.id]
                prompt_counts = await adapter.prompt_counts(plan)
                if set(prompt_counts) != {request.id for request in plan.prepared_requests}:
                    raise ValueError(f"incomplete capacity evidence from {target.engine}")
                counts.append(prompt_counts)
                atomic_json(store.path / f"{target.id}-prompt-counts.json", prompt_counts)
            output_limit = max(r.output_limit for r in plan.prepared_requests)
            context_capacity = capacity(counts, limits, output_limit)
            store.event(
                "prepared",
                context_capacity=context_capacity,
                max_output_tokens=output_limit,
                plan_digest=plan.identity,
            )
            for block, row in enumerate(order):
                for target in row:
                    if runtime_digest(root) != source_identity:
                        raise ValueError("session-bench source changed during execution")
                    adapter = adapters[target.id]
                    progress(f"Pass {block + 1}/{blocks}: {target.id}")
                    async with adapter.launch(
                        context_capacity, plan.parallel_sequences, f"block-{block}"
                    ) as engine:
                        execution = asyncio.create_task(
                            execute(plan, adapter, engine, store, block, records, progress)
                        )
                        exit_watch = asyncio.create_task(engine.process.wait())
                        try:
                            completed, _ = await asyncio.wait(
                                [execution, exit_watch], return_when=asyncio.FIRST_COMPLETED
                            )
                            if exit_watch in completed:
                                raise RuntimeError(f"{target.id} exited during execution")
                            await execution
                        finally:
                            for task in (execution, exit_watch):
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(execution, exit_watch, return_exceptions=True)
                    store.append(
                        "footprints.jsonl",
                        {
                            "target": target.id,
                            "block": block,
                            "baseline_rss_bytes": engine.baseline_bytes,
                            "peak_rss_bytes": engine.peak_bytes,
                        },
                    )
            status = (
                "completed"
                if all(r["observation"]["outcome"] == "valid" for r in records)
                else "failed"
            )
    except asyncio.CancelledError:
        status, error = "cancelled", "run cancelled"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        store.event("error", message=error)
    summary = report.summarize(records, status, command, store.path.name, planned, error)
    footprint_path = store.path / "footprints.jsonl"
    if footprint_path.exists():
        import json

        summary["process_footprints"] = [
            json.loads(line) for line in footprint_path.read_text().splitlines()
        ]
    summary["workload"] = workload
    summary["path"] = str(store.path)
    summary["hardware"] = store.hardware
    summary["thermals"] = thermals.summary
    store.complete(summary, report.markdown(summary))
    return summary
