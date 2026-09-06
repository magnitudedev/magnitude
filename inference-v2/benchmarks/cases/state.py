"""Physical KV mechanisms measured independently of model compute."""

from dataclasses import replace

from benchmarks.contracts import Experiment
from benchmarks.subjects import KVAppend, KVBranch

append_runs = Experiment(
    identity="state.append-runs",
    subject=KVAppend(granularity="runs", prefix_tokens=16384, append_tokens=512),
    characteristic="KV-PREFILL-APPEND",
    measurement_width="component",
    claim="Completed contiguous appends across ten Qwen 35B KV layers, preserving a 16K prefix.",
    warmup=3,
    repetitions=11,
    comparison="Matched state:append_pages isolates call overhead; no throughput parity claim.",
    invariants=(
        "same physical KV geometry",
        "exact prefix and appended values",
        "allocation excluded, writes and completion included",
    ),
)
append_pages = replace(
    append_runs,
    identity="state.append-pages",
    subject=KVAppend(granularity="pages", prefix_tokens=16384, append_tokens=512),
    claim="Explicit per-page reference appends across ten Qwen 35B KV layers with a 16K prefix.",
    comparison="Matched state:append_runs isolates call overhead; not PoC/model throughput parity.",
)
branch = Experiment(
    identity="state.partial-branch",
    subject=KVBranch(page_size=16, prefix_tokens=63, branch_tokens=33),
    characteristic="MECHANISM-KV-LAYOUT",
    measurement_width="integrated",
    claim=(
        "Allocation, immutable partial-prefix retention, two continuations and completed KV "
        "writes. Does not measure model decode or service latency."
    ),
    invariants=(
        "deterministic tensor values",
        "same page/slab/head geometry",
        "fresh arena each repetition",
        "heads-major, four heads, KV width 64, float32",
        "warm after declared warmup; no explicit compilation",
    ),
)
