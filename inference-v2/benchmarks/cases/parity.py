"""Matched plain-generation controls; names identify the executed composition."""

from dataclasses import replace
from typing import Literal

from benchmarks.contracts import Experiment
from benchmarks.subjects import EngineWaves
from magnitude_engine import blueprints as bp

from . import single_session
from .generation import engine_multi_plain, engine_multi_upstream
from .qwen36 import artifact


def custom() -> bp.engine.Engine:
    return replace(
        single_session.engine(),
        generation=bp.generation.Generation(
            target=bp.model.Executor(
                program=bp.model.programs.qwen35.Program(artifact=artifact),
                state=bp.model.state.PagedHybrid(),
            )
        ),
    )


def waves(kind: Literal["custom", "upstream"], context: int, rows: int) -> Experiment:
    case = engine_multi_plain if kind == "custom" else engine_multi_upstream
    assert isinstance(case.subject, EngineWaves)
    assert isinstance(case.subject.engine, bp.engine.Engine)
    composition = replace(
        case.subject.engine,
        scheduler=bp.engine.scheduling.TimeShared(max_active=rows, prefill_tokens=512),
        context_tokens=context + 64,
    )
    return replace(
        case,
        identity=f"engine.qwen36-{kind}-{context}-{rows}-plain-waves",
        subject=replace(case.subject, engine=composition, prompt_tokens=context, rows=rows),
        timeout_seconds=600,
    )


def paged_waves(context: int, rows: int) -> Experiment:
    """Change only the attention consumer in the gathered-attention wave control."""
    case = waves("custom", context, rows)
    assert isinstance(case.subject, EngineWaves)
    composition = case.subject.engine
    assert isinstance(composition, bp.engine.Engine)
    generation = composition.generation
    assert isinstance(generation, bp.generation.Generation)
    target = generation.target
    assert isinstance(target, bp.model.Executor)
    program = target.program
    assert isinstance(program, bp.model.programs.qwen35.Program)
    return replace(
        case,
        identity=f"engine.qwen36-paged-{context}-{rows}-plain-waves",
        subject=replace(
            case.subject,
            engine=replace(
                composition,
                generation=replace(
                    generation,
                    target=replace(
                        target,
                        program=replace(
                            program,
                            attention=bp.model.attention.qwen35.Attention(
                                computation=bp.model.attention.metal.Paged(),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


# Custom program defaults to direct paged attention for decode. Prefill uses
# its declared dense attention operator. Wave controls intentionally retain the
# gathered-attention composition that exposed the original performance gap.
custom_prefill_1k = single_session.prefill(1024, composition=custom(), model="custom-qwen36")
custom_prefill_4k = single_session.prefill(4096, composition=custom(), model="custom-qwen36")
custom_prefill_16k = single_session.prefill(16384, composition=custom(), model="custom-qwen36")
custom_causal_1k = single_session.decode(1024, 4, composition=custom(), model="custom-qwen36")
custom_causal_4k = single_session.decode(4096, 4, composition=custom(), model="custom-qwen36")
custom_causal_16k = single_session.decode(16384, 4, composition=custom(), model="custom-qwen36")
custom_waves_1k_2 = waves("custom", 1024, 2)
upstream_waves_1k_2 = waves("upstream", 1024, 2)
custom_waves_4k_2 = waves("custom", 4096, 2)
upstream_waves_4k_2 = waves("upstream", 4096, 2)
custom_waves_1k_4 = waves("custom", 1024, 4)
upstream_waves_1k_4 = waves("upstream", 1024, 4)
paged_waves_1k_2 = paged_waves(1024, 2)
paged_waves_4k_2 = paged_waves(4096, 2)
paged_waves_1k_4 = paged_waves(1024, 4)
paged_waves_1k_1 = paged_waves(1024, 1)
upstream_waves_1k_1 = waves("upstream", 1024, 1)
