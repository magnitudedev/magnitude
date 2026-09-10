"""GGUF composition for the verbatim session benchmark.

Artifact interpretation belongs to v3; request generation, capacity calculation,
process lifetime, HTTP measurement, validation and reporting are the copied v2
implementation. No MLX artifact alias or second request execution loop is involved.
"""

import argparse
import asyncio
import json
import shlex
from collections.abc import Callable
from pathlib import Path

from benchmark_fixtures import prose
from benchmark_fixtures.prose_history import Prose
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.models.qwen35.artifact import inspect_dense
from magnitude_engine.platform.compiler import compiler_build
from magnitude_engine.platform.measurement import exclusive_measurement
from performance.__main__ import source_identity
from performance.thermals import ThermalRecorder
from session_bench import report
from session_bench.engines.magnitude import Magnitude
from session_bench.models import Artifact, ArtifactFile, Target
from session_bench.results import RunStore, atomic_json
from session_bench.runner import capacity, execute
from session_bench.sessions import Plan
from session_bench.suites import SECTIONS, compile_plan


def gguf_artifact(path: Path) -> Artifact:
    path = path.expanduser().resolve(strict=True)
    before = path.stat()
    artifact = GGUFArtifact(str(path))
    try:
        description = inspect_dense(artifact.directory, artifact.identity)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("GGUF changed during benchmark preparation")
        return Artifact(
            reference=str(path),
            path=path,
            kind="gguf",
            context_limit=description.geometry.context_limit,
            metadata={"architecture": "qwen35", "artifact_identity": artifact.identity},
            files=(
                ArtifactFile(
                    path=path.name,
                    size=after.st_size,
                    sha256=artifact.identity,
                    mtime_ns=after.st_mtime_ns,
                ),
            ),
        )
    finally:
        artifact.close()


async def run(
    root: Path,
    path: Path,
    *,
    sections: tuple[str, ...] = ("single",),
    contexts: tuple[int, ...] = (4096,),
    repeat: int = 1,
    progress: Callable[[str], None] = print,
) -> dict:
    if repeat < 1 or not sections or any(section not in SECTIONS for section in sections):
        raise ValueError("choose supported sections and a positive repetition count")
    if not contexts or any(value < 1 for value in contexts):
        raise ValueError("context checkpoints must be positive")
    path = path.expanduser().resolve(strict=True)
    command = shlex.join(
        [
            "uv",
            "run",
            "--frozen",
            "python",
            "-m",
            "performance.serving",
            "--target",
            str(path),
            "--suite",
            ",".join(sections),
            "--context",
            ",".join(map(str, contexts)),
            "--repeat",
            str(repeat),
        ]
    )
    store = RunStore(
        root,
        command,
        dict(
            target=str(path), sections=sections, contexts=contexts, repeat=repeat, workload="prose"
        ),
    )
    progress(f"Run: {store.path}")
    records: list[dict] = []
    status, error, planned = "failed", None, 0
    thermals = ThermalRecorder(store.path)
    try:
        with exclusive_measurement(), thermals:
            artifact = gguf_artifact(path)
            target = Target(engine="magnitude", reference=str(path))
            adapter = Magnitude(root, target, artifact, store)
            await adapter.prepare()
            source = source_identity(root)
            compiler = compiler_build()
            atomic_json(
                store.path / "engine-build.json",
                dict(source_digest=source, compiler=compiler.model_dump(mode="json")),
            )
            # The copied snapshot covers Python. Preserve native sources and
            # compiler patches too, since they materially define v3 execution.
            for original in (root / "src/magnitude_engine/platform").rglob("*"):
                if original.suffix in (".mm", ".patch") and original.is_file():
                    destination = store.path / "source" / target.id / original.relative_to(root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(original.read_bytes())
            atomic_json(store.path / f"{target.id}-artifact.json", artifact.model_dump(mode="json"))
            atomic_json(store.path / f"{target.id}-runtime.json", adapter.identity)
            text, provenance = await prose.prepare()
            fixtures = Prose(text, provenance)
            async with adapter.context_counter() as counter:
                plan: Plan = await compile_plan(
                    fixtures,
                    fixtures.identity,
                    sections,
                    contexts,
                    counter=counter,
                    sizing_identity=target.id,
                )
            for request in plan.prepared_requests:
                store.append("requests.jsonl", request.model_dump(mode="json"))
            counts = await adapter.prompt_counts(plan)
            if set(counts) != {request.id for request in plan.prepared_requests}:
                raise ValueError("incomplete rendered prompt capacity evidence")
            atomic_json(store.path / f"{target.id}-prompt-counts.json", counts)
            allocated_context = capacity(
                [counts],
                [artifact.context_limit],
                max(r.output_limit for r in plan.prepared_requests),
            )
            planned = len(plan.requests) * repeat
            atomic_json(
                store.path / "plan.json",
                dict(
                    digest=plan.identity,
                    parallel_sequences=plan.parallel_sequences,
                    cache_policy=plan.cache_policy,
                    corpus_digest=plan.corpus_digest,
                    planned_requests=planned,
                    context_capacity=allocated_context,
                    execution_order=[[target.id] for _ in range(repeat)],
                ),
            )
            for block in range(repeat):
                if source_identity(root) != source:
                    raise ValueError("engine or performance source changed during execution")
                async with adapter.launch(
                    allocated_context, plan.parallel_sequences, f"block-{block}"
                ) as engine:
                    await execute(plan, adapter, engine, store, block, records, progress)
                store.append(
                    "footprints.jsonl",
                    dict(
                        target=target.id,
                        block=block,
                        baseline_rss_bytes=engine.baseline_bytes,
                        peak_rss_bytes=engine.peak_bytes,
                    ),
                )
            if source_identity(root) != source:
                raise ValueError("engine or performance source changed during execution")
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
    summary.update(
        workload="prose", path=str(store.path), hardware=store.hardware, thermals=thermals.summary
    )
    store.complete(summary, report.markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--suite", default="single")
    parser.add_argument("--context", default="4096")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    result = asyncio.run(
        run(
            Path(__file__).resolve().parent.parent,
            args.target,
            sections=tuple(args.suite.split(",")),
            contexts=tuple(map(int, args.context.split(","))),
            repeat=args.repeat,
        )
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
