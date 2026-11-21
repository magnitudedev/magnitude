"""One run: template files and catalog artifacts in, sequential engines serving concurrent
simulated sessions, evidence and the result record out.

Every launch's evidence is persisted as it is observed. A failed, cancelled or interrupted
run still writes its summary, report and result record with what it measured.
"""

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from magnitude_fixtures import TemplateFile

from . import record, report
from .adapters import ADAPTERS
from .adapters.base import command as subprocess_command
from .client import Observation, measure
from .harness import HarnessRenderer, rendered_digest
from .models import Acquisition, Artifact, Target, file_hash
from .options import EngineOptions
from .plan import Sized, checked, compile_plan, requested
from .policy import CONTEXT_ALIGNMENT
from .requests import Plan, Request
from .sizing import size
from .store import RunStore, atomic_json

TEMPLATES = "templates"
BUILD = "build.json"


@contextlib.contextmanager
def machine_lock():
    # Process coordination only: one serving benchmark per machine at a time.
    path = Path(tempfile.gettempdir()) / f"magnitude-inference-measurement-{os.getuid()}.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another serving benchmark owns this machine") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def capacity(counts: list[dict[str, int]], limits: list[int], output_limit: int) -> int:
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

    def stream(path: str, fields: dict):
        def record(event: dict) -> None:
            store.record_stream(path, {**fields, **event})
            engine.progress()

        return record

    def save(request: Request, observation: Observation, phase="measured"):
        if (phase, request.id) in recorded:
            return
        recorded.add((phase, request.id))
        engine.finished(request.id)
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
            "template": request.template,
            "lane": request.lane,
            "key": request.key,
            "prompt_sha256": request.prompt_sha256,
            "checkpoint": request.checkpoint,
            "concurrency": request.concurrency,
            "session": request.session,
            "timing_basis": adapter.timing_basis,
            "observation": observation.model_dump(mode="json", exclude_none=True),
        }
        if request.key == "retrieve":
            row["retrieval_total"] = len(request.generation.expected["values"])
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
        engine.started(warmup.id)
        obs = await measure(
            client,
            engine.endpoint,
            engine.model,
            warmup,
            stream("logs/warmup-streams.jsonl", {"target": adapter.target.id, "block": block}),
            extensions=adapter.extensions,
            cancelled=lambda obs: save(warmup, obs, "warmup"),
        )
        save(warmup, obs, "warmup")
        if obs.outcome != "valid":
            raise RuntimeError(f"engine qualification request failed: {obs.outcome}: {obs.error}")

        async def run_request(request, section_started: float):
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
            delay = request.release_ms / 1000 - (time.perf_counter() - section_started)
            if delay > 0:
                await asyncio.sleep(delay)
            store.event(
                "request_started", target=adapter.target.id, block=block, request=request.id
            )
            engine.started(request.id)
            obs = await measure(
                client,
                engine.endpoint,
                engine.model,
                request,
                stream(f"logs/{adapter.target.id}-b{block}-{request.id}.jsonl", {}),
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

        try:
            # Sections are independent schedules and cannot overlap accidentally.
            for section in dict.fromkeys(request.section for request in plan.requests):
                section_started = time.perf_counter()
                async with asyncio.TaskGroup() as group:
                    for request in plan.requests:
                        if request.section == section:
                            tasks[request.id] = group.create_task(
                                run_request(request, section_started)
                            )
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


async def prepare_artifact(target: Target, store: RunStore) -> Artifact:
    """Artifact evidence from a disposable worker, so cancellation stops downloads and hashing."""
    path = store.path / f"{target.id}-artifact.json"
    await subprocess_command(
        [sys.executable, "-m", "magnitude_bench.models", target.model_dump_json(), str(path)],
        store.path,
        store.path / "logs" / f"{target.id}-artifact.log",
    )
    return Artifact.model_validate_json(path.read_text())


def kept(store: RunStore, paths: list[Path]) -> tuple[list[tuple[TemplateFile, Path]], list[dict]]:
    """The run's template files, each kept in the run directory by its digest (the paths the
    harness renders), with their record entries (``templates`` of the result record)."""
    directory = store.path / TEMPLATES
    directory.mkdir(parents=True, exist_ok=True)
    files, entries = [], []
    for path in paths:
        file = TemplateFile.read(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        copy = directory / f"{digest[:16]}.json"
        shutil.copyfile(path, copy)
        files.append((file, copy))
        entries.append({"template": file.template, "context": file.context,
                        "path": str(copy.relative_to(store.path)), "digest": digest})
    return files, entries


async def run(
    store: RunStore,
    templates: list[Path],
    targets: list[Target],
    sections: tuple[str, ...],
    acquisition: Acquisition,
    artifacts: Path,
    build: dict,
    options: EngineOptions,
    command: list[str],
    progress: Callable[[str], None],
) -> dict:
    started_at = record.utc()
    atomic_json(store.path / BUILD, build)
    records: list[dict] = []
    planned = 0
    entries: list[dict] = []
    ordered: list[TemplateFile] = []
    status, error = "failed", None
    adapters = {}  # by target id, as each is prepared
    try:
        files, entries = kept(store, templates)
        ordered = checked([file for file, _ in files])
        paths = {file.template: path for file, path in files}
        atomic_json(
            store.path / "selection.json",
            {
                "templates": [file.template for file in ordered],
                "targets": [target.model_dump(mode="json") for target in targets],
                "sections": sections,
                "engine_options": options.model_dump(mode="json"),
            },
        )
        with machine_lock():
            renderer = HarnessRenderer(
                options.magnitude.harness, artifacts, store.path / BUILD, store.path / "logs"
            )
            limits = []
            for target in targets:
                progress(f"Preparing {target.engine}: {target.reference}")
                artifact = await prepare_artifact(target, store)
                adapter = ADAPTERS[target.engine](target, artifact, acquisition, store, options)
                await adapter.prepare()
                limits.append(artifact.context_limit)
                atomic_json(store.path / f"{target.id}-runtime.json", adapter.identity)
                adapters[target.id] = adapter
            # The catalog GGUF's host-only rendering sizes the run and counts every GGUF
            # engine's prompts; engines serving another form count with their own renderer.
            rendered = [adapter for adapter in adapters.values() if adapter.host_rendered]
            rendering = None
            lines: dict[tuple[str, int], dict] = {}
            if rendered:
                harness = options.magnitude.harness
                if not harness.is_file() or not os.access(harness, os.X_OK):
                    raise ValueError(f"magnitude-harness is unavailable: {harness}")
                rendering = {"harness": str(harness), "harness_sha256": file_hash(harness)}
            sizing = "magnitude-harness" if rendered else targets[0].engine
            names = ", ".join(file.template for file in ordered)
            progress(f"Sizing {names} with {sizing}'s rendering")
            pairs = requested(ordered, sections)
            if rendered:
                lines = await renderer.render([(file, paths[file.template]) for file in ordered],
                                              store.path)
                sized = {(file.template, lane): renderer.sized(file, lane, lines[(file.template, lane)])
                         for file, lane in pairs}
            else:
                async with adapters[targets[0].id].counter() as lengths:
                    conversations = await size(
                        [(file.sessions[lane], file.context) for file, lane in pairs], lengths
                    )
                sized = {(file.template, lane): Sized(conversation, None)
                         for (file, lane), conversation in zip(pairs, conversations, strict=True)}
            plan = compile_plan(ordered, sections, sized)
            for request in plan.prepared_requests:
                store.append("requests.jsonl", request.model_dump(mode="json"))
            blocks = options.schedule.blocks(len(targets))
            order = [
                targets[i % len(targets) :] + targets[: i % len(targets)] for i in range(blocks)
            ]
            planned = len(plan.requests) * len(targets) * blocks
            host_counts = (
                {request.id: lines[(request.template, request.lane)]["generation_prefix_tokens"]
                 for request in plan.prepared_requests}
                if rendered else None
            )
            counts = []
            for target in targets:
                adapter = adapters[target.id]
                prompt_counts = (
                    host_counts if adapter.host_rendered else await adapter.prompt_counts(plan)
                )
                counts.append(prompt_counts)
                atomic_json(store.path / f"{target.id}-prompt-counts.json", prompt_counts)
            output_limit = max(r.output_limit for r in plan.prepared_requests)
            context_capacity = capacity(counts, limits, output_limit)
            atomic_json(
                store.path / "plan.json",
                {
                    "digest": plan.identity,
                    "sized_by": sizing,
                    "rendering": rendering,
                    "parallel_sequences": plan.parallel_sequences,
                    "cache_policy": plan.cache_policy,
                    "context_capacity": context_capacity,
                    "planned_requests": planned,
                    "execution_order": [[target.id for target in row] for row in order],
                },
            )
            store.event(
                "prepared",
                context_capacity=context_capacity,
                max_output_tokens=output_limit,
                plan_digest=plan.identity,
            )
            for block, row in enumerate(order):
                for target in row:
                    adapter = adapters[target.id]
                    progress(f"Pass {block + 1}/{blocks}: {target.id}")
                    await measure_block(adapter, plan, context_capacity, block, store, records,
                                        progress)
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
    summary = report.summarize(records, status, " ".join(command), store.path.name, planned, error)
    footprint_path = store.path / "footprints.jsonl"
    if footprint_path.exists():
        summary["process_footprints"] = [
            json.loads(line) for line in footprint_path.read_text().splitlines()
        ]
    allocation_path = store.path / "allocations.jsonl"
    if allocation_path.exists():
        summary["expert_banks"] = report.expert_banks(
            [json.loads(line) for line in allocation_path.read_text().splitlines()],
            {target: adapter.expert_bank for target, adapter in adapters.items()},
        )
    tuning_path = store.path / record.TUNING_TRACE
    if tuning_path.exists():
        if health := report.tuning_health(tuning_path):
            summary["tuning_health"] = health
    summary["templates"] = [file.template for file in ordered]
    summary["key"] = ordered[0].sessions[0].key if ordered else None
    summary["path"] = str(store.path)
    host = store.path / "host.json"
    if host.is_file():
        summary["host"] = json.loads(host.read_text())
    store.complete(summary, report.markdown(summary))
    write_result(store, started_at, build, acquisition, artifacts, entries, records, options,
                 any(target.engine == "magnitude" for target in targets), command)
    return summary


async def measure_block(
    adapter, plan: Plan, context_capacity: int, block: int, store: RunStore, records: list[dict],
    progress: Callable[[str], None],
) -> None:
    label = f"block-{block}"
    async with adapter.launch(context_capacity, plan.parallel_sequences, label) as engine:
        execution = asyncio.create_task(
            execute(plan, adapter, engine, store, block, records, progress)
        )
        exit_watch = asyncio.create_task(engine.process.wait())
        try:
            completed, _ = await asyncio.wait(
                [execution, exit_watch], return_when=asyncio.FIRST_COMPLETED
            )
            if exit_watch in completed:
                if engine.retired is not None:
                    raise RuntimeError(f"{adapter.target.id} watchdog: {engine.retired}")
                raise RuntimeError(f"{adapter.target.id} exited during execution")
            await execution
            allocation = await adapter.memory_observation(engine)
            if allocation is not None:
                store.append(  # once the pass ended; the samples while it served are `allocations.jsonl`
                    "pass-allocations.jsonl",
                    {"target": adapter.target.id, "block": block, "observation": allocation},
                )
        finally:
            for task in (execution, exit_watch):
                if not task.done():
                    task.cancel()
            await asyncio.gather(execution, exit_watch, return_exceptions=True)
    store.append(
        "footprints.jsonl",
        {
            "target": adapter.target.id,
            "block": block,
            "baseline_rss_bytes": engine.baseline_bytes,
            "peak_rss_bytes": engine.peak_bytes,
        },
    )


def write_result(
    store: RunStore,
    started_at: str,
    build: dict,
    acquisition: Acquisition,
    artifacts: Path,
    templates: list[dict],
    records: list[dict],
    options: EngineOptions,
    magnitude: bool,
    command: list[str],
) -> None:
    """The contract result record; host facts and the thermal trace are added around the job."""
    rows = [record.serving_row(row) for row in records if row["phase"] == "measured"]
    measurements: dict = {"serving": rows}
    traces = {}
    if magnitude:
        drafting = record.drafting(rows, "magnitude")
        if drafting is not None:
            blocks = {row["block"] for row in records if row["target"] == "magnitude"}
            drafting["rounds"] = record.block_draft_rounds(store.path / "logs", blocks)
            measurements["drafting"] = drafting
        trace = store.path / record.TUNING_TRACE
        if trace.is_file():
            budget, fresh = options.magnitude.tuning_seconds, options.magnitude.fresh_tuning
            try:
                measurements["tuning"] = record.tuning(trace, budget, fresh)
            except ValueError as error:
                # A load cut off mid-tuning (a failed or cancelled run) keeps its trace.
                outcome = {"error": str(error)}
                measurements["tuning"] = {"budget_seconds": budget, "fresh": fresh,
                                          "outcome": outcome}
            traces["tuning"] = record.TUNING_TRACE
    atomic_json(
        store.path / "result.json",
        {
            "schema": record.SCHEMA,
            "tool": "bench",
            "started_at": started_at,
            "finished_at": record.utc(),
            "build": build,
            "model": acquisition.model,
            "artifacts": json.loads(artifacts.read_text()),
            "templates": templates,
            "rendered_digest": rendered_digest(store.path),
            "host": {},
            "command": command,
            "measurements": measurements,
            "traces": traces,
        },
    )
