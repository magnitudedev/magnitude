"""Four workload/composition quadrants, plus independent-row execution controls."""

from dataclasses import replace
from typing import Literal

from benchmarks.contracts import Experiment
from benchmarks.subjects import EngineWaves
from benchmarks.subjects.generation.blueprint import GenerationDecode
from magnitude_engine import blueprints as bp

from .qwen36 import engine as speculative_engine


def decode(
    *, speculative: bool, rows: int, context: int = 128,
    execution: Literal['shared', 'independent'] = 'shared',
) -> Experiment:
    assert isinstance(speculative_engine.generation, bp.generation.Generation)
    composition = replace(
        speculative_engine,
        generation=(speculative_engine.generation if speculative else replace(
            speculative_engine.generation, method=bp.generation.methods.Plain(),
        )),
        context_tokens=context + rows + 64,
        scheduler=bp.engine.scheduling.TimeShared(max_active=rows),
    )
    return Experiment(
        identity=(
            f'generation.qwen36-{context}-{rows}-{"mtp" if speculative else "plain"}-{execution}'
        ),
        subject=GenerationDecode(
            engine=composition, prompt_tokens=context, output_tokens=32, rows=rows,
            execution=execution,
        ),
        characteristic='MECHANISM-GENERATION-COMPOSITION',
        measurement_width='integrated',
        claim='Completed fixed-context decode, excluding scheduler and prompt service.',
        comparison='Shared execution versus independent requests with the same target and method.',
        invariants=(
            'independent greedy target oracle', 'unequal row context lengths',
            'same linked checkpoints for every sample', 'same token allowance',
            'device completion inside timing',
        ),
        warmup=1, repetitions=3, timeout_seconds=300, run_class='diagnostic',
    )


single_plain = decode(speculative=False, rows=1)
multi_plain = decode(speculative=False, rows=2)
single_speculative = decode(speculative=True, rows=1)
multi_speculative = decode(speculative=True, rows=2)
multi_plain_independent = decode(speculative=False, rows=2, execution='independent')
multi_speculative_independent = decode(speculative=True, rows=2, execution='independent')


def engine_waves(*, speculative: bool, rows: int) -> Experiment:
    from .qwen36 import waves

    assert isinstance(waves.subject, EngineWaves)
    benchmark = decode(speculative=speculative, rows=rows)
    assert isinstance(benchmark.subject, GenerationDecode)
    composition = benchmark.subject.engine
    assert isinstance(composition, bp.engine.Engine)
    composition = replace(composition, context_tokens=64, scheduler=bp.engine.scheduling.TimeShared(
        max_active=rows, max_queued=rows, prefill_tokens=64,
    ))
    return replace(
        waves,
        identity=f'engine.qwen36-{rows}-{"mtp" if speculative else "plain"}-waves',
        subject=replace(waves.subject, engine=composition, rows=rows),
    )


engine_single_plain = engine_waves(speculative=False, rows=1)
engine_multi_plain = engine_waves(speculative=False, rows=2)
engine_single_speculative = engine_waves(speculative=True, rows=1)
engine_multi_speculative = engine_waves(speculative=True, rows=2)


def upstream_waves(rows: int) -> Experiment:
    from .qwen36 import artifact

    baseline = engine_waves(speculative=False, rows=rows)
    assert isinstance(baseline.subject, EngineWaves)
    assert isinstance(baseline.subject.engine, bp.engine.Engine)
    composition = replace(
        baseline.subject.engine,
        generation=bp.generation.Generation(target=bp.model.auto(artifact)),
    )
    return replace(
        baseline, identity=f'engine.qwen36-{rows}-upstream-waves',
        subject=replace(baseline.subject, engine=composition),
    )


engine_single_upstream = upstream_waves(1)
engine_multi_upstream = upstream_waves(2)
