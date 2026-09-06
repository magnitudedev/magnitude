"""Owned Qwen prefill, replay and generation on shared context fixtures."""

from typing import Literal

from benchmarks.contracts import Experiment
from benchmarks.subjects.context.blueprint import ContextWorkload
from magnitude_engine import blueprints as bp

from .qwen36 import artifact


def workload(
    context: int,
    mode: Literal["prefill", "replay", "generate"] = "replay",
    fixture: Literal["prose.moby-dick", "tools.bfcl"] = "prose.moby-dick",
    measured_tokens: int = 32,
) -> Experiment:
    engine = bp.engine.Engine(
        generation=bp.generation.Generation(
            target=bp.model.Executor(
                program=bp.model.programs.qwen35.Program(artifact=artifact),
                state=bp.model.state.PagedHybrid(),
            )
        ),
        scheduler=bp.engine.scheduling.TimeShared(max_active=1, prefill_tokens=512),
        prefixes=bp.engine.prefixes.Radix(
            retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=0),
        ),
        context_tokens=context + measured_tokens + 8192,
    )
    return Experiment(
        identity=f"model.qwen36-{mode}-{fixture}-at-{context}-n{measured_tokens}",
        subject=ContextWorkload(
            engine=engine,
            artifact=artifact.path,
            fixture=fixture,
            context_tokens=context,
            mode=mode,
            measured_tokens=measured_tokens,
        ),
        characteristic="MODEL-QWEN35-FORWARD",
        measurement_width="integrated",
        claim=f"Owned {mode} on pinned {fixture}; includes model transaction/completion costs.",
        comparison="Matched fixture and checkpoint; content is separate from the execution mode.",
        invariants=(
            "fixed checkpoint",
            "prefix preparation and restoration excluded",
            "committed work count checked",
            "fixture provenance recorded",
        ),
        warmup=3,
        repetitions=11,
        timeout_seconds=900,
        run_class="diagnostic",
    )


replay_4k = workload(4096)
replay_16k = workload(16384)
replay_64k = workload(65536)
generate_4k = workload(4096, "generate")
generate_64k = workload(65536, "generate")
prefill_4k = workload(4096, "prefill", measured_tokens=512)
prefill_64k = workload(65536, "prefill", measured_tokens=512)
tools_replay_4k = workload(4096, fixture="tools.bfcl", measured_tokens=16)
tools_replay_64k = workload(65536, fixture="tools.bfcl", measured_tokens=16)
tools_generate_4k = workload(4096, "generate", fixture="tools.bfcl")
tools_generate_64k = workload(65536, "generate", fixture="tools.bfcl")
