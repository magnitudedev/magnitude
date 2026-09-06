# Generic MLX-VLM

## Scope

**An owned executor around an upstream model, with an independently comparable
boundary between them.** IDs follow [component identification](../../components.md).
Each node below has a definition here; references identify comparison controls,
not additional production dependencies.

The current path supports standalone text modules exposed by MLX-VLM, resident
float or affine weights, and explicitly adapted native caches. It exposes logits,
not intermediate features or media conditioning. Upstream model availability does
not imply that its cache, batching or modality capabilities are supported here.

## Assembly

```text
MODEL:EXECUTOR:MAG:UPSTREAM
├── MODEL:LOADING:MAG:UPSTREAM             construction, not per-token work
├── MODEL:FORWARD:VLM:STANDARD             neural execution delegated as a unit
└── STATE:CHECKPOINTS:MAG:NATIVE           cache adaptation and transactions
```

The executor supplies positions, invokes the language forward and publishes its
state. The state adapter owns reservations and checkpoints; upstream owns neural
semantics. This generic tree stops at the upstream forward because its internal
architecture varies with the artifact. A claim about one of its internal blocks
must identify that artifact's block and reference explicitly.

## Components

### `MODEL:EXECUTOR:MAG:UPSTREAM`

- **Contract / implementation:** Tokens and logical starting state become requested
  logits and advanced state. Own position binding, compatible batch assembly,
  completion roots and resource lifetime around the two runtime children above.
  No added whole-model compilation; unrequested logits remain unevaluated where
  their computation is exclusive to those outputs.
- **References / tests:** Invoke a separately loaded stock MLX-VLM model with the
  same artifact, tokens, positions and state. Compare logits and subsequent state
  behavior for single requests, batch changes and restored prefixes.
- **Performance / bounds:** Model latency is upstream forward plus exposed adapter
  work. Separate position construction, batch/cache movement, submission and
  completion from neural execution; do not double-count overlapping work. Stock
  performance is the integration target. Derive further headroom from the actual
  model's traffic, arithmetic and dependencies rather than a generic token ceiling.

### `MODEL:LOADING:MAG:UPSTREAM`

- **Contract / implementation:** Resolve upstream configuration and language module,
  validate tensor layout, materialize supported text weights and own their budgeted
  lifetime. Peer modality weights are excluded. Loading failures release resources.
- **References / tests:** Compare tensor values, names, encodings and model arguments
  with direct upstream loading. Exercise unsupported layouts and partial failures.
- **Performance / bounds:** Measure startup latency, bytes read and peak live memory.
  Model storage reads, format conversion and device materialization by their actual
  dependencies and bandwidths. Count retained weights once, plus conversion scratch
  and overlapping allocations. These are `MODEL:LOADING/LAT` and
  `MODEL:LOADING/MEM`; cold and warm loading are separate operating points.

### `MODEL:FORWARD:VLM:STANDARD`

- **Contract / implementation:** The pinned upstream standalone language model
  consumes tokens/native caches and produces logits with updated caches. Its blocks,
  kernels and numerical conventions remain upstream-owned.
- **References / tests:** Direct invocation is the reference for our integration.
  It cannot independently validate its own equations: use an independent model
  implementation or mathematical block oracle when those equations are in question.
- **Performance / bounds:** Instantiate the artifact's layer graph, active weight
  traffic, attention geometry and recurrent work. Prefill and decode have different
  reuse. The bound follows that graph and hardware resources; upstream timing is
  evidence of attainment, not a ceiling. No single bound applies to all VLM models.

### `STATE:CHECKPOINTS:MAG:NATIVE`

- **Contract / implementation:** Wrap supported upstream cache objects with reserve,
  begin, advance, checkpoint and restore behavior. Preserve logical positions and
  state isolation; account for replacement peaks before execution. Shared by generic
  targets and compatible attached heads, with model-specific capacity geometry.
- **References / tests:** Compare independently advanced upstream caches and exact
  logical checkpoint contents. Exercise window crossings, multi-input extensions,
  rejected suffixes, batch changes and budget failure before mutation.
- **Performance / bounds:** Track retained bytes, replacement/copy bytes, transient
  peaks and checkpoint/restore latency. Append caches grow with history; recurrent
  state is fixed-size; rotating caches retain a window but a query may need
  `window + query_width - 1` visible keys. Derive movement costs from actual arrays
  and bandwidth; stable batches should not reconstruct complete histories per token.
  `STATE:CHECKPOINTS/MEM` scores retained footprint and
  `STATE:CHECKPOINTS/RESTORE` scores readiness after restoration, including deferred
  repair. Transient peaks and checkpoint creation costs remain visible constraints
  and enclosing-workload costs.

## Performance composition

Every contract in this tree resolves to its [ceiling binding](../../performance/catalog.md#generic-upstream-contracts).
The [common definition](../../performance.md) provides an optimistic theoretical
bound per declared dimension, independent of source/variant. References and current
implementation costs diagnose gaps; they do not limit that bound. Parent accounting
allows fusion and shared-data reuse before counting unavoidable demands.

Executor and forward use `/EXEC`, one execution-efficiency percentage each.
Loading declares `MODEL:LOADING/LAT` and `MODEL:LOADING/MEM`: startup latency and
peak loading footprint can trade off through staging/conversion concurrency.
Native checkpoints declare `STATE:CHECKPOINTS/MEM` and `STATE:CHECKPOINTS/RESTORE`:
retained footprint and restoration time can trade off through checkpoint retention.
The [catalog definitions](../../performance/catalog.md#dimension-definitions) fix each
metric and boundary. Tree values are samples at the selected workload, not universal
scores for every upstream model.

The neural model and integration have separate cost models. A slow executor can be
localized to the forward or to native state/batching overhead. An upstream kernel
replacement belongs in an explicit architecture assembly, not a hidden exception
inside this pass-through binding.

Use [optimization](../optimization.md) to combine child costs. Native memory
reservations are capacity obligations, not evidence that those bytes move on every
forward. Loading belongs in startup measurements, not steady decode throughput.

## Qualification

Current assessments follow the [evidence/reset rules](../../performance.md#evidence-and-current-assessments):
any implementation change makes its scores and affected parent scores `unmeasured`.
Historical observations remain tied to their original fingerprints and operating points.

These IDs describe existing responsibilities; they do not assert universal upstream
parity. The September 6 native Gemma comparison established matching logits for the
recorded 1K/16K/32K cases after window-accounting correction. It did not establish
rotating-cache batching, broad artifact support or a speedup. New source revisions
and model/cache combinations require their own evidence keyed to the same IDs.

Evidence: `sessions/26-09-06/evidence/cycle-006/comparison.json` and
`qualification.manifest.json` in that directory, relative to the monorepo root.
