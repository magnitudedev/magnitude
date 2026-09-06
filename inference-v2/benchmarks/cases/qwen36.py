"""Pinned local Qwen target/head compositions; importing cases never loads their weights."""

from pathlib import Path

from benchmarks.contracts import Experiment
from benchmarks.subjects import EngineWaves, ModelPrefill
from magnitude_engine import blueprints as bp

_cache = Path.home() / ".cache/huggingface/hub"
artifact = bp.model.artifacts.Local(
    path=str(
        _cache
        / "models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots"
        / "38740b847e4cb78f352aba30aa41c76e08e6eb46"
    )
)
head_artifact = bp.model.artifacts.Local(
    path=str(
        _cache
        / "models--mlx-community--Qwen3.6-35B-A3B-MTP-bf16/snapshots"
        / "e931b93eed744eae16049d4ebeddf636ef5b90f2"
    )
)


def prefill(input_tokens: int) -> Experiment:
    engine = bp.engine.Engine(
        generation=bp.generation.Generation(
            target=bp.model.Executor(
                program=bp.model.programs.qwen35.Program(artifact=artifact),
                state=bp.model.state.PagedHybrid(),
            )
        ),
        scheduler=bp.engine.scheduling.TimeShared(max_active=2),
        prefixes=bp.engine.prefixes.Radix(
            retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=0),
        ),
        context_tokens=input_tokens + 8,
    )
    return Experiment(
        identity=f"model.qwen36-prefill-{input_tokens}",
        subject=ModelPrefill(engine=engine, prefix_tokens=0, input_tokens=input_tokens),
        characteristic="MECHANISM-PREFILL-CHUNK-EFFICIENCY",
        measurement_width="integrated",
        claim=(
            "Completed prefill chunk cost for one Qwen3.6 35B Q4 sequence at zero prior context. "
            "Excludes loading, restore, scheduling and HTTP. Repeatability and finite state "
            "are checked; independent neural accuracy/BFCL are separate gates. No ceiling claim."
        ),
        warmup=1,
        repetitions=5,
        run_class="diagnostic",
        invariants=(
            "same locked local artifact",
            "same prefix checkpoint per sample",
            "exact input count",
            "repeatable continuation logits",
            "finite continuation logits",
            "completed forward and accepted state inside timing",
        ),
    )


prefill_16 = prefill(16)
prefill_32 = prefill(32)
prefill_64 = prefill(64)
prefill_128 = prefill(128)
prefill_256 = prefill(256)
prefill_512 = prefill(512)

reader = bp.resources.io.PositionalReader(workers=4)
program = bp.model.programs.qwen35.Program(
    artifact=artifact,
    attention=bp.model.attention.qwen35.Attention(computation=bp.model.attention.mlx.Gathered()),
    reader=reader,
)
head = bp.model.programs.mtp.Head(artifact=head_artifact, target=program, reader=reader)
engine = bp.engine.Engine(
    generation=bp.generation.Generation(
        target=bp.model.Executor(program=program, state=bp.model.state.PagedHybrid()),
        method=bp.generation.methods.MTP(
            drafter=bp.model.Executor(
                program=head,
                state=bp.model.state.Native(source=head),
            )
        ),
    ),
    scheduler=bp.engine.scheduling.TimeShared(max_active=2, max_queued=2, prefill_tokens=64),
    prefixes=bp.engine.prefixes.Radix(
        retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=4),
    ),
    context_tokens=40,
    output_capacity=4,
)
waves = Experiment(
    identity="engine.qwen36-35b-mtp-two-waves",
    subject=EngineWaves(
        engine=engine,
        prompt_text="Write a Python function that adds two numbers.\n\ndef add(a, b):",
        prompt_tokens=16,
        output_tokens=16,
        rows=2,
    ),
    characteristic="MECHANISM-ENGINE-SERVICE",
    measurement_width="workload",
    claim=(
        "Real Qwen3.6 35B Q4/MTP concurrent generation then complete prefix reuse. Includes "
        "engine service and delivery; excludes loading, tokenization, HTTP and BFCL. "
        "Not performance acceptance or a competitor comparison."
    ),
    warmup=1,
    repetitions=3,
    comparison="single-candidate characterization only",
    invariants=(
        "same exact local artifacts",
        "greedy output matches plain-generation oracle",
        "two cold then two prefix-warm requests",
        "actual prompt count checked",
        "same reusable arena and emptied prefix store at repetition start",
    ),
)
