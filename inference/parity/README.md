# ICN primitive parity

This directory defines narrow, attributable comparisons against pinned llama.cpp. It does not benchmark complete chat responses, HTTP behavior, scheduling policy, or server throughput.

Every claim belongs to one of two categories:

- **Correctness:** the native reference and production Rust/ICN path produce the same primitive output or state transition.
- **Performance:** both paths perform the same observed work inside the same timing boundary with comparable cost.

An upstream test passing qualifies the reference build. It does not establish ICN parity by itself.

## What compares to what

There are two paired comparison boundaries:

1. **Official tool ↔ production probe.** A pinned machine-readable llama.cpp tool is the reference and the matching `icn-llamacpp` operation is the candidate. P1–P4 use this boundary so their work and timing remain defined by `llama-bench`.
2. **Native oracle ↔ production probe.** A thin C++ process calls the pinned public C API or common helper while the candidate calls the finalized safe Rust binding through production-owned `icn-llamacpp` code. Correctness differentials and custom P6–P8 microbenchmarks use this boundary.

`upstream-test` and `upstream-qualification` are one-sided reference/build checks, not paired parity. P0 is intentionally one-sided.

The runner compares immutable evidence outside both engines:

```text
case -> reference adapter -> reference evidence
     -> candidate adapter -> candidate evidence

reference evidence + candidate evidence + declared comparator -> comparison
```

Cases contain neutral inputs, never machine-local paths, C structs, Rust types, HTTP payloads, or hidden scheduler policy. The adapters resolve verified artifacts and translate the neutral contract to their respective interface.

## Strict JSONL producers

The oracle and candidate probe use schema-v1 JSON Lines on stdin/stdout. Each non-empty request contains `schemaVersion`, `caseId`, `operation`, and an `input` object; unknown envelope fields and unsupported versions are rejected. Every request produces one response that echoes the case and operation and contains either evidence or a typed error.

Input lines are capped at 16 MiB. The runner bounds execution time, stdout, and stderr; the candidate adapter contains panics as structured errors; diagnostics may not contaminate protocol stdout. The transport binary contains no primitive implementation—those operations live in `icn-llamacpp` and use the production safe bindings.

Before running cases, the runner calls `protocol.describe`. The live candidate operation set must exactly equal the static capabilities in [`producers.toml`](producers.toml) plus `protocol.describe`. A stale capability in either direction fails preflight.

## Layout

- [`cases/`](cases/) contains one neutral descriptor per isolated claim. `active` means both adapters exist and activation verification passed; `planned` is a visible, non-claiming gap; `disabled` requires an explicit reason.
- [`fixtures/`](fixtures/) contains small deterministic inputs. Every case reference carries a SHA-256.
- [`models/registry.toml`](models/registry.toml) contains accepted external artifacts. URLs are acquisition locations; bytes and SHA-256 define identity.
- [`profiles/`](profiles/) selects cases and execution policy without changing workload or thresholds.
- [`producers.toml`](producers.toml) declares candidate transport, decoder, limits, and exact static operations; it contains no developer-local binary path.
- [`upstream/`](upstream/) declares pinned source/build lanes, bounded tests/tools, and finalized binding surfaces.
- [`oracle/`](oracle/) is the thin C++ reference process.
- [`schemas/`](schemas/) is the versioned neutral record contract.

Generated builds, model caches, and evidence live outside this directory under ignored paths such as `target/parity-models/` and `results/parity/<run-id>/`.

## Primitive boundaries

The descriptor is authoritative. Planned cases are excluded by profiles that select `statuses = ["active"]`.

| Primitive | Descriptor and isolated claim | State |
| --- | --- | --- |
| C0 native baseline | `correctness.baseline.focused-upstream-tests`: one-sided focused upstream tests qualify the pinned native build before differential evidence is interpreted. | Active |
| C1 configuration | `correctness.config.context-defaults`: construct one context and compare public effective configuration; no decode. | Active |
| C2 metadata | `correctness.metadata.stable-fields`: load a model and compare stable model/vocabulary metadata; no context. | Active |
| C3 tokenize | `correctness.tokenizer.unicode-special-flags`: compare token IDs for fixed UTF-8 bytes and explicit BOS/special-token policy. | Active |
| C3 token-to-piece | `correctness.tokenizer.token-to-piece-unicode-bytes`: compare per-token piece bytes and their byte concatenation; this is not a bulk detokenization claim. | Active |
| C4 chat template | `correctness.chat-template.chatml-basic`: compare the prepared prompt representation only; parsing, tokenization, and decode are excluded. | Active |
| C5 grammar/schema | `correctness.grammar.json-schema-boolean-object`: compare exact JSON-Schema-to-grammar conversion. | Active |
| C6 top-k transform | `correctness.sampler.top-k-three`: compare only transformed candidate ordering and values. | Active |
| C6 seeded selection | `correctness.sampler.distribution-four-seeded`: separately compare normalized probabilities and the selected token. | Active |
| C7 reasoning budget | `correctness.reasoning-budget.controllable-force`: compare bounded public `CommonSampler` force and last-token observations around a forced ending. | Active |
| C8 output parser | `correctness.parser.content-only-unicode`: compare content-only parser snapshots, semantic deltas, and final messages across declared byte partitions; no model generation. | Active |
| C9 batch/decode | `correctness.decode.single-batch-logits`: compare one explicit CPU-resident decode status and selected logits; no tokenization or sampling. | Active |
| C10 KV/state | `correctness.state.remove-and-redecode`: compare a bounded sequence removal, re-decode status, state, and selected logits in a CPU-resident full-attention context. | Active |
| C11 abort | `correctness.cancellation.decode-abort`: compare only the outcome class of a pre-signalled decode abort. | Active |
| C11 recovery | `correctness.cancellation.decode-abort-recovery`: remove the callback, clear logical memory, and compare a recovery decode plus selected logits. | Active |
| C12 mtmd | `correctness.mtmd.preprocess-structure`: projector capability and media-preprocess structure. | Disabled: no accepted model/projector/media tuple, qualified producer feature, or three-role artifact injection. |
| C13 fidelity | `correctness.fidelity.tiny-corpus-perplexity`: fixed-corpus likelihood scoring. | Disabled: no production scoring API; a parity-only scorer would test scaffolding instead. |
| P0 backend ops | `performance.backend-ops.native-perf-qualification`: one-sided native backend-operation corpus; never a binding-overhead comparison. | Active |
| P1 prompt throughput | `performance.llama-bench.prompt-512`: official `llama-bench` prompt decode versus the same observed candidate work. | Active |
| P2 generation throughput | `performance.llama-bench.generation-128`: official single-token decode loop; tokenization, sampling, and piece conversion are excluded. | Active |
| P3 prompt + generation | `performance.llama-bench.prompt-generation-128-32`: the official combined prompt/decode interval. | Active |
| P4 context depth | `performance.llama-bench.context-depth-128`: generation from the same prepared context depth. | Active |
| P5 multi-sequence batch | `performance.batched-bench.two-sequence-independent-prompts`: matched independent-prompt workload shape using producer-valid synthetic tokens. | Planned: no production candidate operation yet. |
| P6 sampler | `performance.sampler.top-k-large-vector`: copy, apply, reset, and semantic fold over the same deterministic vector. | Active |
| P7 chat preparation | `performance.chat-template.chatml-basic`: repeated common-chat preparation with construction outside the timer. | Active |
| P8 parser | `performance.parser.content-only`: reusable content-only parse plus semantic projection; preparation, generation, and serialization are excluded. | Active |

## Record contracts

[`case.schema.json`](schemas/case.schema.json) separates neutral inputs, requirements, invocations, comparator, and performance timing boundaries. Comparator JSON pointers are relative to the evidence `output` value.

[`evidence.schema.json`](schemas/evidence.schema.json) records outcome, observed work, operation output, all raw samples, source/binary/build identity, host/devices, artifact digests, and effective configuration. Byte-bearing fields use integer arrays rather than lossy text.

[`comparison.schema.json`](schemas/comparison.schema.json) distinguishes:

- `pass` or `fail`: work/configuration matched, so the declared comparison is valid;
- `invalid`: work, provenance, or timing boundaries differed, so no parity conclusion exists;
- `skipped`: a declared requirement was unavailable.

Performance is never “timing only.” The runner first verifies identical observed work. Differential microbenchmarks also exact-compare their declared semantic output before evaluating a duration ratio. This prevents a fast no-op or a subtly different operation from passing.

There is no fuzzy text comparator and no aggregate parity score.

## Fairness and profiles

- `pr` runs active correctness/reference qualification cases; performance is disabled.
- `diagnostic` runs one pair of every selected active two-sided correctness/performance case without claiming a controlled or exclusive host. It is a functional protocol, capability, and work-boundary smoke; its recorded timings are uncontrolled, non-qualification, and non-gating evidence.
- `native-diagnostic` runs only the one-sided C0 and P0 native checks on an uncontrolled developer host. It verifies the pinned build/test plumbing and preserves P0 timing, but it makes no candidate-parity or controlled-performance claim and does not gate on performance.
- `nightly` selects active primitives, alternates paired engine order, requires exclusive-device attestation but not a controlled-host attestation, and treats performance as exploratory rather than gating.
- `release` requires both controlled-host and exclusive-device attestations and gates on performance.

Both sides use the same resolved engine configuration. Performance runs are serial, preserve every raw repetition, and require their declared correctness prerequisite to pass for the same model/backend. A planned, unselected, failed, invalid, or skipped prerequisite cannot authorize a performance claim.

Throughput bounds are minimum candidate/reference ratios; duration bounds are maximum ratios. Faster candidates are recorded as improvements. Results from uncontrolled developer machines remain diagnostic.

C9–C11 deliberately use CPU-resident tiny-model contexts: zero GPU layers, K/Q/V and operation offload disabled, flash attention off, and one main/batch thread. Zero GPU layers alone is not a residency guarantee. Accelerator and additional state-architecture variants require separate matched cases.

P8 includes safe-wrapper semantic projection in its timer while fixture I/O, chat preparation, parser construction, generation, and serialization remain outside. It can expose real Rust ownership/copy overhead; a native-kernel-only parser benchmark would be a different claim.

## Models and offline staging

The initial accepted model is `stories15m-q4-0`, identified by SHA-256 `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`. Fetch it explicitly into a dedicated cache:

```bash
cargo run -p icn-parity -- models --root parity fetch \
  --id stories15m-q4-0 --model-root target/parity-models
```

Execution never downloads. The runner verifies selected registry files before invocation. The C0 wrapper stages a create-only private copy into the fresh upstream build, requires the cache and build tree to be disjoint, rejects hard-link aliasing, and suppresses CTest's network-capable download setup.

Model-backed cases expand over the selected registry list and filter by `valid_for` and architecture tags, so compatible models do not require duplicated descriptors. Artifact-specific token vectors remain explicitly tied to their model. C12 cannot activate until a compatible accepted model, projector, and media fixture are all registered and verified.

## Validation and builds

Run from `inference/`:

```bash
cargo run -p icn-parity -- validate --root parity
cargo run -p icn-parity -- list --root parity --profile pr
cargo run -p icn-parity -- run --root parity --profile pr
```

Reference and candidate executables are accepted only through fresh content-addressed manifests:

```bash
bun run scripts/build-reference.ts --target focused-tests --target oracle --target llama-bench
bun run scripts/build-candidate.ts --reference-manifest <cargo-equivalent-reference.json>
```

The builders verify pinned source identity, inventory inputs before and after compilation, and bind each executable to its digest. A descriptor becomes active only after its required reference and production candidate operations pass the paired smoke.
