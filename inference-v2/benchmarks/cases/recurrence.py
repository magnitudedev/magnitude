"""Fixed verification work and cold prefill-shape compilation controls."""

from dataclasses import replace

from benchmarks.contracts import Experiment
from benchmarks.subjects import Recurrence, RecurrenceShapes
from magnitude_engine import blueprints as bp

metal = Experiment(
    identity="operator.delta-owned",
    subject=Recurrence(update=bp.model.recurrence.metal.Delta(), tokens=3),
    characteristic="RECURRENT-STATE-ADVANCE",
    measurement_width="component",
    claim="Completed Metal recurrence at three tokens; no model/acceptance claim.",
    warmup=3,
    repetitions=11,
    comparison="Matched recurrence:mlx; development characterization only.",
    invariants=(
        "same deterministic prepared inputs and initial state",
        "completed output and state inside timing",
        "library numerical oracle outside timing",
        "resident prepared inputs; reset initial recurrence state",
        "fresh lazy invocation; warmed Metal compilation",
    ),
)
mlx = replace(
    metal,
    identity="operator.delta-library",
    subject=Recurrence(update=bp.model.recurrence.mlx.Delta(), tokens=3),
    claim="Completed pinned MLX-LM recurrence at three tokens; no model/acceptance claim.",
    comparison="Matched recurrence:metal; development characterization only.",
)
shapes = Experiment(
    identity="operator.delta-prefill-lengths",
    subject=RecurrenceShapes(update=bp.model.recurrence.metal.Delta()),
    characteristic="RECURRENCE-COLD-SHAPE-COST",
    measurement_width="component",
    claim=(
        "First completed Metal calls at distinct prefill lengths, including invocation/completion. "
        "Independent MLX-LM oracles precede timing. Not a pure compiler timer or model benchmark."
    ),
    warmup=0,
    repetitions=1,
    run_class="diagnostic",
    invariants=(
        "fresh benchmark process",
        "no Metal recurrence warmup",
        "same head/width/dtype geometry",
        "distinct long input lengths",
        "library output and state oracle for every length",
    ),
)
specialized_shapes = replace(
    shapes,
    identity="operator.delta-specialized-prefill-lengths",
    subject=RecurrenceShapes(update=bp.model.recurrence.metal.Delta(specialize_prefill=True)),
    claim="Control for recurrence:shapes, specializing each length. No engine/acceptance claim.",
)

# Long recurrence is a separate operating point from speculative verification.
# Both cases retain the same prepared input/state and numerical oracle contract.
metal_prefill = replace(
    metal, identity="operator.delta-owned-prefill-512",
    subject=Recurrence(update=bp.model.recurrence.metal.Delta(), tokens=512),
    claim="Completed 512-input owned recurrence at Qwen geometry, excluding model preparation.",
)
mlx_prefill = replace(
    mlx, identity="operator.delta-library-prefill-512",
    subject=Recurrence(update=bp.model.recurrence.mlx.Delta(), tokens=512),
    claim="Completed 512-input MLX-LM recurrence at Qwen geometry, excluding model preparation.",
)
