# Tile sizing and autotuning on M-series Macs

Research note, 7 September 2026. This is a proposal grounded in public documentation,
upstream source, and the current `inference-v2` implementation, not an adopted design
or a report of measured speedups. No tuning benchmarks were run for this note.

The recommended direction is **operation-specific candidate schedules, filtered by
correctness and device limits, selected using measurements, and reused through a
persistent machine-specific selection cache**. Ship useful defaults; perform bounded
local tuning during preparation or idle periods. Ordinary inference should execute
an already selected schedule.

## 1. What a tile is choosing

A tile assigns a subset of an operation to an execution scope and determines which
values stay live there. There are several distinct choices:

| Choice | Main effect |
|---|---|
| Output tile | Reuse inputs across independent outputs; determines accumulator storage and number of workgroups |
| Reduction tile | How much input to stage or process before advancing; affects barriers, pipelining and live storage |
| SIMD-group ownership | Which threads cooperate, share values, or reduce together |
| Threadgroup geometry | How SIMD groups are packaged and synchronized |
| Reduction partitioning | Creates parallel partial results, requiring combination and a numerical contract |
| Traversal order | Changes locality between neighboring workgroups |

These must be chosen together. A thread count alone is not a schedule, and changing
only a launch size can make a kernel incorrect if its indexing assumes another shape.

## 2. Deriving the tradeoffs

Consider C[M,N] = A[M,K] B[K,N]. A threadgroup produces an m-by-n output tile while
processing k reduction elements per iteration. Assuming each input tile is loaded
once at the memory boundary under discussion, equal input element size s, and
ignoring output traffic and edge padding:

```
work per iteration       = 2 m n k FLOPs
input bytes per iteration = s k (m + n)
arithmetic intensity      = 2 m n / [s (m + n)] FLOPs/byte
```

This is a derivation for an idealized schedule, not a prediction of actual DRAM
traffic: caching, repeated loads, quantization metadata and spills change the result.
For square tiles with FP16 inputs:

| Output tile | Input arithmetic intensity | FP32 output accumulator footprint across the group |
|---|---:|---:|
| 32 × 32 | 16 FLOPs/byte | 4 KiB |
| 64 × 64 | 32 FLOPs/byte | 16 KiB |
| 128 × 128 | 64 FLOPs/byte | 64 KiB |

The accumulator column is logical storage distributed across participating threads,
not a Metal threadgroup-memory allocation or an exact compiled register count.
Doubling both output dimensions doubles input reuse but quadruples accumulator
storage and reduces the number of output tiles by roughly four.

Increasing k cancels out of this intensity formula. A larger reduction tile can
amortize barriers and loop overhead or expose independent loads, but it does not
automatically increase reuse. If both inputs are explicitly staged, their footprint
is approximately s k (m+n), multiplied by the number of simultaneously live buffers.

Three additional constraints determine whether reuse translates into speed:

1. **Enough independent work.** Output workgroups number roughly
   ceil(M/m) × ceil(N/n) × batch. A large tile can leave a large GPU underfilled.
   Splitting K creates more workgroups but introduces partial-result traffic and
   combination. The break-even point depends on the entire region.
2. **Enough concurrent work per core.** More live values can reduce concurrent
   execution or cause spill traffic. Occupancy is a means of hiding latency, not the
   objective; a lower-occupancy kernel can win by doing less memory traffic.
3. **Useful work and locality.** Tail padding wastes work. Adjacent lanes should
   access memory efficiently, and the mapping must match packing and layout.
   Cross-SIMD exchanges and barriers have costs that a FLOP/byte model omits.

A rough diagnostic model is dispatch cost plus the larger of compute time and
memory-transfer time, with synchronization and dependency stalls where they are not
hidden. It is not a rigorous sum or a universal performance ceiling. Analyze each
relevant memory level and instruction class; quantized kernels also unpack, convert,
address scales and perform integer work.

Apple's scaling guidance explicitly distinguishes bandwidth and compute limits,
emphasizes sufficient workgroups, and recommends avoiding unnecessarily large
threadgroups. [Apple: Scale compute workloads across Apple GPUs](https://developer.apple.com/videos/play/wwdc2022/10159/).

### Decode and prefill are different regimes

For a single matrix-vector product whose weights stream from memory once, the
weight-only intensity is approximately 2/s_w FLOPs per byte. This is 1 for FP16 and
4 for ideal packed 4-bit weights, before metadata and dequantization costs. More
output channels per tile can reuse the activation vector, but do not make each
weight useful for multiple tokens. A larger query/request tile can do that when
rows actually share weights. In MoE, reuse depends on requests selecting the same
expert; nominal batch size is an incomplete predictor.

For attention, several queries or GQA heads can reuse each KV load. But each needs
its own output accumulator and softmax state. Single-query decode offers less query
reuse and can need context partitioning to expose parallelism. Longer prefill has
more query parallelism and may benefit from matrix-oriented implementations.
FlashAttention develops the memory-traffic argument; FlashAttention-2 shows that
work partitioning and reduced inter-warp communication also matter. Their algorithmic
lessons transfer; their NVIDIA tile constants do not establish Apple optima.
[FlashAttention](https://arxiv.org/abs/2205.14135),
[FlashAttention-2](https://arxiv.org/abs/2307.08691).

## 3. What is specific to Apple silicon

- **Legal limits come from the device and compiled pipeline.** Query execution
  width, maximum threads for that pipeline, and device threadgroup-memory capacity.
  A device-wide 1,024-thread ceiling is not proof that every compiled kernel permits
  that size. [Apple: calculating threadgroups](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes).
- **Documented family ceilings are not performance targets.** Apple's current
  tables list 32 KiB threadgroup memory and 1,024 threads for the relevant Apple GPU
  families. Our 24 KiB attention budget is a chosen budget below a ceiling.
  [Apple: Metal limits](https://developer.apple.com/metal/limits/).
- **M3 changed resource behavior.** Apple family 9 dynamically allocates register
  storage over a shader's lifetime and shares on-chip capacity across several memory
  uses. A static register-budget equation is consequently an imperfect performance
  model, particularly across generations.
  [Apple: M3 and A17 Pro architecture](https://developer.apple.com/videos/play/tech-talks/111375/).
- **M5 introduces another implementation opportunity.** TensorOps can use GPU neural
  accelerators on M5 and optimized shader paths on older chips. Availability also
  depends on the OS/API version, especially for newer datatypes. Candidate families
  should be capability-gated; hardware portability does not imply identical optimal
  schedules. [Apple: M5 ML workloads](https://developer.apple.com/videos/play/tech-talks/111432/).
- **Compiler behavior matters.** Dynamic indexing of thread-local arrays can spill;
  register allocation has granularity. Counting source variables does not reveal
  exact resource use. Inspect representative variants with Xcode's GPU profiler.
  [Apple: Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/).

Use exact device identity and available capabilities for local results. Use GPU family
defaults as starting points, not evidence of optimality on every base, Pro, Max or
Ultra configuration. Include the software/compiler environment in cache validity.
MLX exposes device information, but its documented dictionary is backend-dependent;
do not assume it exposes all pipeline limits or occupancy information.
[MLX device information](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.device_info.html).

## 4. What existing tuning systems teach us

| System | Mechanism | Relevance to Magnitude |
|---|---|---|
| Triton autotune | User-defined configurations, tuning keys, pruning, measurement and optional disk caching | Best small-system template for our parameterized Metal kernels |
| tinygrad BEAM | Searches transformations, compiles and measures candidates, caches schedules | A real Metal-capable example of machine-local search, but operates on tinygrad's own compiler representation |
| TVM Ansor / MetaSchedule | Generates broader schedules, uses cost models and budgets, stores measured results | Useful when developing a compiler/search space; much larger scope than tuning our existing kernels |
| Kernel Tuner | Parameter search with constraints, correctness checks, timing and persistent records | Useful search/measurement design; its documented backends do not provide a native Metal backend |
| llama.cpp Metal tuner | Offline device-local sweeps produce dispatch tables | Particularly relevant precedent for Apple inference defaults and shape buckets |
| MLX Metal GEMM | Architecture-, shape-, dtype- and layout-dependent dispatch heuristics | Demonstrates that maintaining a finite set of specialized schedules is practical on our existing backend |

Triton explicitly allows early legality pruning and ranking with a performance model,
then benchmarking only a shortlist. Its decorator runs a kernel repeatedly during
tuning, and provides reset/restore hooks for modified tensors. Upstream Triton's
supported-platform list does not make it a drop-in Metal solution. Experimental
Metal forks exist, but their existence is not broad M-series qualification.
[Triton autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html),
[upstream compatibility](https://github.com/triton-lang/triton#compatibility),
[experimental Metal fork](https://github.com/ben594/triton-metal).

tinygrad supports M1+ Metal devices. Its beam search explores schedule actions,
deduplicates compiled programs, measures candidates with early stopping and persists
the selected transformations. This supports the feasibility of local search without
requiring us to adopt its tensor runtime.
[tinygrad runtimes](https://docs.tinygrad.org/runtime/),
[beam-search implementation](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/codegen/opt/search.py).

Ansor combines a hierarchical search space with evolutionary search and a learned
cost model. MetaSchedule exposes separate builders, runners, cost models, databases
and global/per-task trial budgets. My recommendation is to borrow bounded search and
record reuse first; introduce a learned ranker only if measured search costs justify
it. [Ansor paper](https://arxiv.org/abs/2006.06762),
[MetaSchedule API](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html).

Kernel Tuner supports constrained configurations, supplied reference outputs and
persistent measurement caches. Metal integration would require an adapter or host
wrapper rather than its normal native backend path.
[Kernel Tuner API](https://kerneltuner.github.io/kernel_tuner/stable/user-api.html),
[backends](https://kerneltuner.github.io/kernel_tuner/stable/backends.html).

llama.cpp's current Metal tuner targets flash-attention vector configurations. Its
full documented sweep takes hours; it uses baseline anchors to detect timing drift
and conservative bucket acceptance. Numerical validation is a separate step, with
documented coverage limitations. Runtime selection tries exact device entries,
family entries, then baseline. These are useful patterns, not a ready-made tuner
for our kernels. [Tuner workflow](https://github.com/ggml-org/llama.cpp/blob/master/tools/tuning/README.md),
[dispatch selection](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-tuning.cpp).

The MLX version pinned by this project, 0.32.2, already selects GEMM tiles using
architecture and workload conditions. It has separate matrix paths with different
geometry, including neural-accelerator paths. This is evidence for candidate families,
not a claim that MLX is autotuning our custom kernels.
[MLX 0.32.2 GEMM dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/matmul.cpp).

## 5. JIT compilation, autotuning and caching are separate

**JIT compilation** turns one selected specialization into executable code.
**Autotuning** executes multiple legal alternatives to choose one.
**Selection caching** remembers that decision. **Executable caching** avoids compiling
the chosen variant again. Persisting a tile choice alone does not eliminate cold
compilation in a new process.

MLX recommends creating a custom kernel once and reusing it; template parameters
support specialization. Our shared runtime already caches kernel construction in
process. It does not contain a tuning-result database.
[MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).

Metal offers binary archives to avoid applicable runtime compilation. Their existence
does not mean our Python wrapper automatically persists custom kernels this way;
that integration should be investigated only if measured startup cost warrants it.
[Apple: Metal binary archives](https://developer.apple.com/documentation/metal/metal-binary-archives).

For one tuning key, let C be incremental tuning cost and delta_t the saving per
future call. Tuning pays back after roughly C / delta_t uses. For example, a
hypothetical 0.2-second tuning cost and 5-microsecond saving need 40,000 calls.
Count compatible invocations across repeated layers and sessions, not merely tokens.
These numbers illustrate the calculation; they are not timings measured on our code.

Recommended runtime behavior:

1. Load compatible saved selections during model preparation.
2. On a miss, immediately use a qualified family default.
3. Queue worthwhile misses for bounded calibration during preparation or GPU-idle
   periods. A background thread using a busy GPU still competes with inference.
4. Start with a small shortlist, perhaps 4–8 candidates per frequently used regime,
   and a total model-preparation budget. These are proposed policy values to measure.
5. Persist only meaningful, repeatable gains. Prewarm and bind selected variants
   before entering the steady execution path.

A warm call should need only a prebound variant or small in-memory lookup. Do not
read a tuning database, compile all candidates, synchronize the GPU or explore on
every token. Changing selections must account for compiled graph capture; a Python
lookup updated after capture may not affect an already compiled graph.

## 6. The smallest useful architecture

Three responsibilities suffice:

- **Operation:** defines acceptable arithmetic, rounding, outputs and state effects.
- **Schedule family:** exposes legal alternatives, derives matching launch geometry,
  supplies resource estimates and provides a fallback.
- **Selection:** chooses from those alternatives using device/workload information
  and qualified measurements; stores evidence and the chosen configuration.

Keep family-specific knowledge local. Share legality checks that really are common,
measurement infrastructure, budget handling, cache validity and reporting. Existing
performance tooling should own measurement campaigns; production should consume
compact selections without importing benchmark or theoretical-model machinery.

A selection key needs the numerical family/version, source dependencies, relevant
shape regime, dtype/encoding/group size, layout and mask mode, device identity and
software environment. Some dimensions require exact values; others can use measured
buckets. Benchmark dtype, head width, query count, context range, batch geometry,
and relevant routing patterns rather than model names. Exact runtime dimensions
remain operands or launch parameters even when selection uses a bucket.

Start broad shape buckets only where supported by evidence. Test their endpoints and
known dispatch boundaries; do not tune every new context length. Avoid data-dependent
host synchronization merely to compute a better tuning key. Candidate compilation
and cache growth also require bounds.

For a small pruned space, direct enumeration with staged timing is sufficient. With
larger spaces, first use a good default plus structured neighbors, preserving coupled
choices; one-knob-at-a-time greedy search can miss interactions. Learned ranking or
Bayesian search is a later option, not the foundation of the initial architecture.

## 7. Numerical qualification limits the search space

Floating-point reductions are not associative. Varying split-K, attention partitions,
SIMD reduction width, accumulation dtype or math mode can change results. Such changes
are eligible only under the operation's established contract. A numerical tolerance
must not be relaxed simply because a candidate is fast.

The existing design requires attention's logical summary structure and per-query
visibility to be independent of peer history and page-table capacity. Therefore,
selection based on batch geometry may rearrange independent outputs but must not
silently change a row's arithmetic. Keep logical reduction boundaries explicit,
even if physical execution combines them within a threadgroup.

There is a specific audit item in the current implementation: the caller-supplied
`covered` value controls family choice and subdivision; callers derive it from
maximum lengths or allocated coverage. Establish which choices preserve the required
invariants before exposing them to tuning. This observation identifies a qualification
question, not a demonstrated numerical failure.

Never measure alternative recurrence/state-write kernels by repeatedly mutating a
live session. Use isolated state snapshots. Qualification needs independent reference
outputs, masks/tails, page relocation, batch regrouping and continuation behavior.

## 8. Applying this to our kernels

| Family | Useful candidate axes | Constraints and evidence |
|---|---|---|
| Short-query attention | Queries and shared heads per SIMD group, key lookahead, physical summary placement, qualified partition choices | Register/live-value growth, scratch, masks, per-row summary contract; time partials and combine together |
| Expert gate/up and down | Output channels per SIMD group, SIMD groups per threadgroup, shared activation loading, qualified row reuse | Packing/group alignment, expert reuse distribution, required casts; time gate/up, activation and down together |
| Normalization | SIMD groups per row, values per lane, multiple independent rows per group | Complete reduction ownership, supported widths and numerical reduction contract |
| Recurrence | Independent heads/value channels per group | Preserve token order and accepted-prefix recovery; avoid live-state tuning |

Our existing attention arrays make the resource tradeoff explicit. With W=32 lanes,
HP heads, QT queries and KT keys, their logical float-slot count per lane is roughly:

```
HP * QT * (DK/W + DV/W + 2) + KT * (DK/W + DV/W)
```

This excludes compiler temporaries, addresses and later combination values. The key
lookahead's 32-float allowance is only one contribution, not a cap on total registers.
For multiple subchunks, shared scratch is `4 * HG * NC * (DV+2)` bytes. These formulas
can prune or rank choices, but they cannot predict exact occupancy.

The expert kernel's current four channels per SIMD group is embedded in Metal loops
as well as Python launch planning. Tuning it requires a coherent parameterized family;
changing the Python `tile = 8` constant alone is insufficient. The current custom
attention path supports 1–8 query tokens, so large-prefill optimization is a different
implementation-family investigation.

## 9. Measurement and a useful first experiment

Use evaluated inputs and completed GPU work: MLX is lazy, so graph-construction time
is not kernel time. Separate compile/startup costs, device execution and end-to-end
latency. [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

Warm variants before timing, compare candidates in interleaved order against a
baseline, repeat finalists, and require a gain larger than measurement uncertainty.
Use representative weight/KV working sets: repeatedly timing one small cached tensor
can choose a schedule that loses while streaming real model layers. Record power
mode and timing drift; use one measurement workload per Mac at a time. Confirm final
choices through the enclosing region and normal compiled model path.

The first experiment should parameterize one high-impact existing family, retain the
current schedule as control, and measure a small legal shortlist across representative
M-series generations and GPU sizes. Attention and expert projections are plausible
starting points; existing model sensitivity measurements should choose the priority.

Record four outcomes: numerical qualification, steady execution gain, incremental
cold tuning/compilation time, and selection hit rate on realistic shape sequences.
If a small shortlist reliably beats defaults and amortizes its cost, add persistent
local calibration. If it does not, improved offline defaults may be sufficient.
There is currently no evidence in this research establishing a particular winning
tile size, tuning budget or end-to-end speedup for Magnitude.
