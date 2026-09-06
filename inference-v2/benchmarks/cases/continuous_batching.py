"""Identical cold prompt waves through Magnitude and stock MLX-VLM scheduling."""
from dataclasses import replace
from typing import Literal

from benchmarks.subjects import EngineWaves
from benchmarks.subjects.upstream.blueprint import UpstreamWaves
from magnitude_engine import blueprints as bp

from .parity import paged_waves, waves
from .qwen36 import artifact


def cold_waves(engine: Literal["upstream", "custom", "mlx-vlm"], context: int, rows: int):
    case = paged_waves(context, rows) if engine == "custom" else waves("upstream", context, rows)
    assert isinstance(case.subject, EngineWaves)
    subject = case.subject
    composition = subject.engine
    assert isinstance(composition, bp.engine.Engine)
    if engine == "mlx-vlm":
        selected = UpstreamWaves(
            artifact=artifact, prompt_text=subject.prompt_text, prompt_tokens=context,
            output_tokens=subject.output_tokens, rows=rows,
        )
    else:
        selected = replace(
            subject, prefix_reuse=False,
            engine=replace(composition, prefixes=bp.engine.prefixes.Radix(
                retention=bp.engine.prefixes.LeastRecentlyUsed(max_entries=0),
            )),
        )
    return replace(
        case, identity=f"continuous-batching.{engine}-{context}-{rows}-cold",
        subject=selected, characteristic="WORKLOAD-COLD-CONTINUOUS-BATCHING",
        comparison="Match prompt/output digests across engines before comparing elapsed time.",
        claim=(
            "Two cold waves of simultaneous equal-context requests, greedy fixed output, "
            "resident target, no speculation, retained prefixes, tool grammar or HTTP. "
            "Compare complete workload and first-token latencies; phase definitions differ."
        ),
        invariants=(
            "same natural prompt tokenization and output allowance",
            "all requested outputs complete", "repeatable output tokens",
            "engine output also checked against independent request execution",
            "no retained-prefix reuse", "completion inside timing",
        ),
    )


native_1k_4 = cold_waves("upstream", 1024, 4)
custom_1k_4 = cold_waves("custom", 1024, 4)
vlm_1k_4 = cold_waves("mlx-vlm", 1024, 4)
native_4k_4 = cold_waves("upstream", 4096, 4)
custom_4k_4 = cold_waves("custom", 4096, 4)
vlm_4k_4 = cold_waves("mlx-vlm", 4096, 4)
