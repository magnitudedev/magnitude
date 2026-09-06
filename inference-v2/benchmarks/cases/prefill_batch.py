"""Matched prompt computation: same total tokens, shared versus independent execution."""
from dataclasses import replace
from typing import Literal

from benchmarks.contracts import Experiment
from benchmarks.subjects import ModelPrefill
from magnitude_engine import blueprints as bp

from .single_session import engine


def prefill(width: int, execution: Literal["shared", "independent"]) -> Experiment:
    return Experiment(
        identity=f"model.qwen36-prefill-4x{width}-{execution}",
        subject=ModelPrefill(
            engine=replace(engine(), context_tokens=1024, scheduler=bp.engine.scheduling.TimeShared(
                max_active=4,
            )),
            prefix_tokens=0, input_tokens=width, rows=4, execution=execution,
        ),
        characteristic="MECHANISM-PROMPT-BATCHING",
        measurement_width="integrated",
        claim=(
            "Completed prefill of four equal-width prompts with the same bound upstream program. "
            "State creation, capacity reservation, and continuation probes are outside timing. "
            "Shared execution includes physical batch formation; no scheduler or HTTP is involved."
        ),
        comparison="Compare matching widths across shared and independent execution.",
        warmup=1, repetitions=3, timeout_seconds=300, run_class="diagnostic",
        invariants=(
            "same input tokens and total work",
            "declared physical batching required",
            "every committed state length checked",
            "finite repeatable continuation logits at each execution geometry",
            "completion inside measurement",
        ),
    )


shared_32 = prefill(32, "shared")
independent_32 = prefill(32, "independent")
shared_128 = prefill(128, "shared")
independent_128 = prefill(128, "independent")
shared_512 = prefill(512, "shared")
independent_512 = prefill(512, "independent")
