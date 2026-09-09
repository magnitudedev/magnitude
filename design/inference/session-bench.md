---
applies_to:
  - inference-v2/src/session_bench/**
  - inference-v2/tests/session_bench/**
  - inference-v2/pyproject.toml
  - inference-v2/session-bench.md
  - inference-v2/session-bench-runtimes/**
---

# Session bench

Session bench is development tooling that measures inference serving using simulated agent sessions.
It is not an agent evaluator, the official BFCL leaderboard, or the application's ICN lifecycle.
It calls disposable serving processes directly and does not change first-party product ownership.
The Python source lives under inference-v2/src/session_bench and the command is session-bench.

## Selection and identity

Commands select models, engines, sections, context targets and optional repetitions. No experiment
files are authored or loaded. Model aliases live exclusively in inference-v2/models.local.json,
gitignored and resolved relative to that file. Aliases map MLX/GGUF representations to local paths
or pinned Hub references. No home-directory alias configuration or checked-in model registry exists.
Magnitude means the Python engine; upstream llama.cpp, MLX-VLM and oMLX are comparison engines.
The old Magnitude llama.cpp/ICN engine is not a target. Engine source directory names are not public IDs.

## Work and measurement

Pinned and hash-verified BFCL interactions, pinned prose, and versioned RULER-derived synthetic
retrieval recipes build deterministic requests. Tool and prose workloads construct canonical completed
history. Retrieval fixes facts and questions independently of sizing and uses complete distractor
records to resize context; evidence depth moves the fact block without changing its answers.
Retrieval checkpoints preserve requested order and repeats, and never replay probe answers into
later snapshots. Request bodies, expectations and dependencies are shared across targets. Observed output
never becomes the input to subsequent shared requests. Context targets are approximate input sizes;
terminal engine counts are authoritative measurements. Session, independent concurrency, fork,
concurrency pressure and memory sections describe offered traffic; evidence explicitly distinguishes
history sharing from actual retained-prefix reuse. No retention claim follows from session shape alone.

Tool requests have a fixed 32,768 completion-token allowance, prose 256 and retrieval 1,024, with no
CLI or environment override. Engine capacity must cover rendered inputs plus that full allowance within model limits.
Shared capacity rounds up to 256-token allocation boundaries.
Preparation tokenization is capacity evidence, never measured token evidence. Length termination is
truncation for tools and retrieval, including parseable partial answers. Prose may terminate normally
at its full output budget; ending for length before that budget is truncation.
Sampling is greedy, seed 42 where supported, and model-selected thinking is disabled.

Adapters own engine preparation, launch, readiness and cleanup; shared code owns session scheduling,
HTTP/SSE, semantic validation and reporting. Preparation installs frozen dependencies and obtains
artifacts before timed work. Measurement is offline with verified source, artifact and runtime identity.
Only one benchmark owns the machine's managed benchmark process lifetime at once. Targets run
sequentially, with balanced fresh-process passes and cache-disjoint warmup. Within a target, requests
follow the declared dependency graph and release schedule. Prefix policy is recorded and verified;
the initial cross-engine baseline disables retained prefixes and does not claim cached-session results.

Terminal usage and native timing counters must agree. No native time or token count is fabricated
from text or client latency. Different artifacts/templates or observed prompt counts prevent strict
comparison. Stock MLX-VLM emission timing is labeled and excluded from cross-engine native phase
ratios. Tool calls are matched as a multiset with complete assignment across overlapping alternatives.
Retrieval answers are exact JSON string mappings with duplicate keys rejected. Report whole-answer
accuracy and partial field accuracy separately; additional keys fail whole-answer equality. Retrieval
qualification uses a separate short fixture. Synthetic fixtures retain upstream reference, local recipe,
seed, sizing identity, requested/actual lengths, record positions and content digest. Rebuilding a size
with the same fixture and renderer reproduces the same input, independently of prior preparations.
Different generated prose or tool encodings are real serving-work differences, not evidence of
slower neural execution. Use fixed-work component/engine controls from the
[benchmark hierarchy](benchmarking.md) to isolate those costs.

## Persistence and lifecycle

Each measured invocation creates inference-v2/runs/session-bench/<UTC-id>/ before preparation. It saves
the public invocation and a shell-quoted reproduction command expanded to immutable engine/artifact
pairs. Reproduction does not read old result schemas or depend on aliases; local paths still require
the recorded bytes, and exact historical reproduction requires recorded code/dependencies/hardware.
Relevant source snapshots, artifact hashes, corpus identity, runtime versions and effective policy
accompany results. The shared benchmark hardware record is captured once before preparation,
retained in the initial run record and final summary, and identified in the Markdown report.
The shared benchmark temperature recorder spans preparation through engine cleanup;
its per-sensor Celsius trace and summary remain attached when serving observations
are imported into the performance store. Reports label temperature aggregates as
whole-run evidence, rather than attributing them to individual requests or targets.
Only relevant source is captured, never credentials or the full environment.

Events and completed request results are appended and flushed during execution. Final summaries and
Markdown are written atomically. Partial failures and cancellation preserve evidence and trigger
bounded process-tree retirement. Inspection does not initialize engines. Interrupted runs cannot
appear completed. Historical formats need no migration or execution support; raw evidence and
self-contained reports remain accessible.

Reports show failure/invalid/truncated counts and metric denominators. Context scaling may summarize
semantically invalid but protocol-complete results, separately from correctness. Retrieval includes
both correct and incorrect protocol-complete answers in latency summaries for every section; all
recorded retrieval requests enter accuracy denominators, with unscored failures earning zero.
Other tool sections require both gates. Protocol/transport errors, timeouts, cancellation and truncation
never enter performance summaries. Warmup and qualification are separate from measured observations.

## Acceptance

Tests demonstrate deterministic shared sessions, local alias resolution, alias-independent saved
commands, immutable output policy, input-plus-output capacity checks, fragmented SSE handling,
terminal consistency, non-greedy semantic matching, cancellation cleanup and persistent partial
results. Retrieval tests additionally demonstrate reversible resizing, stable facts, depth control,
strict answer matching, no answer leakage and failure-inclusive accuracy denominators.
Adapter integration is qualified against the actual serving interface, not an invented
benchmark-only inference implementation. Unsupported capabilities fail explicitly.
