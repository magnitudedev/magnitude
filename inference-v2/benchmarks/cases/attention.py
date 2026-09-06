"""Matched completed attention at one Qwen 35B layer's geometry."""

from dataclasses import replace

from benchmarks.contracts import Experiment
from benchmarks.subjects import Attention
from magnitude_engine import blueprints as bp

metal = Experiment(
    identity="operator.attention-metal-16k",
    subject=Attention(computation=bp.model.attention.metal.Paged(), prefix_tokens=16384),
    characteristic="PAGED-DECODE-ATTENTION",
    measurement_width="component",
    claim="KV append plus direct paged attention for one Qwen 35B layer at 16K context.",
    warmup=3,
    repetitions=15,
    comparison="Matched attention:gathered; development characterization, not whole-engine parity.",
    invariants=(
        "identical BF16 queries and KV",
        "same physical page placement",
        "append and completion inside timing",
        "rounded FP32 attention oracle outside timing",
    ),
)

gathered = replace(
    metal,
    identity="operator.attention-gathered-16k",
    subject=Attention(computation=bp.model.attention.mlx.Gathered(), prefix_tokens=16384),
    claim="Characterize KV append plus gathered SDPA for one Qwen 35B layer at 16K context.",
    comparison="Matched attention:metal; development characterization, not whole-engine parity.",
)
