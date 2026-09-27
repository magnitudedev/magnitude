# Exact solver evaluation lab

This CPU-only harness tests the mathematical solver independently of Seismic,
languages, accelerators and hardware cost models. Its invented integer costs and
task durations define synthetic problems; they do not predict native kernel time.

Generated data, plots and logs under `results/`, CP-SAT output under `evidence/`,
and Python caches are Git-ignored. Markdown reports remain versioned. Raw-artifact
links in reports refer to local output; use the recorded commands to regenerate it.

## Reproduce a small exactness cohort

```sh
cargo test --manifest-path inference-v4/solver/lab/Cargo.toml
cargo run --release --manifest-path inference-v4/solver/lab/Cargo.toml -- generate --suite oracle-small --seeds 0..9 --out /tmp/solver-lab/oracle
cargo run --release --manifest-path inference-v4/solver/lab/Cargo.toml -- verify --input /tmp/solver-lab/oracle --reference exhaustive --assignments 1000000 --seconds-per-case 10
```

Seeds `0..9` are the published development cohort. Use held-out seeds `100..109`
after tuning; seed ranges are inclusive. Suite commands default to `0..9`, while a
single-family generation defaults to seed 0. Generators use specified SplitMix64
arithmetic, and saved instances include every constraint and objective term.
The model fingerprint is diagnostic FNV-1a over its JSON, never a proof or memo key.

Verification reports `agreement: null` when a reference or the tested solver did
not complete. This is censored verification, not a passing exactness claim.
Coverage-gap fixtures instead require an incomplete result because their
unresolved branch can beat the available witness. An incorrect exact answer exits
with an error. Every solver and optional CP-SAT optimal witness is re-evaluated
against the original immutable model.

## Families and their structural questions

| Family | Structure | Independent reference |
| --- | --- | --- |
| `independent` | n independent d-way alternatives | Sum of local minima |
| `chain` | Adjacent representation costs | O(n d²) dynamic program |
| `separator` | Sliding factor bags of width+1 | Tiny exhaustive enumeration |
| `shared-producer` | One explicit preparation versus separate occurrences; capacity can forbid sharing | Closed form |
| `process-plan` | An alternative activates further finite decisions | Bottom-up minimum of generated tree |
| `schedule` | Optional modes, short precedence chains, a shared cumulative resource | Tiny bounded enumeration or CP-SAT |
| `repeated` | Identical independent serial bodies, shared implementation decision | Scaled sum of minima |
| `coverage-gap` | Competitive, declared missing construction | Must remain incomplete |
| `pipeline` | Preparation and consumers with shared production, retained storage and concurrency | Tiny enumeration or CP-SAT |

The conditional fixture has A=2+9+9=20 and B=8+3+4=15. It rejects the tempting
but invalid policy of choosing the cheapest producer independently. Sharing with
two consumers costs 6+1+1=8; disallowing sharing restores 6+6+1+1=14.

Pipeline choices include group width g, a window q dividing the input length K,
traversal batch width u, and retain/recompute policy. Each window has preparation
duration `overhead+q`; each group consumes `1+actual_group_width*q` ticks, including
an explicit final group. The transfer resource has capacity 1; compute capacity
is `n`; retained data reserves `q*element_bytes` from preparation completion until
last use. Each batch pays an explicit compute overhead activity. Thus changing
g/q/u changes finite task graphs and resource interactions while preserving the
total represented quantities. These are generic hypothetical tasks.

All competing reservations belong to one resource constraint. Independent
optimization never merges distinct preparation occurrences. Repeated-body cost
multiplication is only used for explicitly additive independent bodies; it is
not an approximation of parallel makespan. Large explicitly expanded pipelines
have a 4096-activity construction cap and report a construction error rather
than secretly dropping alternatives.

## Scaling and ablations

```sh
cargo run --release --manifest-path inference-v4/solver/lab/Cargo.toml -- bench --suite structure-v1 --seeds 0..9 --policy dfs --seconds-per-case 60 --memory-mib 2048 --repeats 5 --out /tmp/solver-lab/structure
cargo run --release --manifest-path inference-v4/solver/lab/Cargo.toml -- compare --input /tmp/solver-lab/structure/instances --out /tmp/solver-lab/comparison --policies dfs,best-first,no-cache,no-decompose,weak-bounds --seconds-per-case 60 --repeats 5
cargo run --release --manifest-path inference-v4/solver/lab/Cargo.toml -- report --input /tmp/solver-lab/comparison --out /tmp/solver-lab/report
```

Generate once, then use identical instance files for every policy. The five
policies differ only in traversal, caching, decomposition or bound strength;
their completion standard is identical. `structure-v1` varies n=8,32,128,512;
d=2,4,8,16,64; constructed width=1,2,4,6,8; and repetition counts up to one million.
The generator parameter is constructed bag width, not measured minimum treewidth.
`scheduling-v1` varies explicit task counts; `pipeline-v1` increases quantities
and storage interactions. These are one-axis sweeps, not a full Cartesian product.

For a small controlled timing experiment:

```sh
cargo run --release --manifest-path inference-v4/solver/lab/Cargo.toml -- generate --family chain --n 32 --d 4 --seeds 100..109 --out /tmp/solver-lab/held-out
cargo run --release --manifest-path inference-v4/solver/lab/Cargo.toml -- bench --input /tmp/solver-lab/held-out --seconds-per-case 5 --memory-mib 512 --repeats 5 --out /tmp/solver-lab/held-out-results
```

Generator options are `n,d,width,depth,repeat,capacity,horizon,outputs,input-length,
overhead,element-bytes`. Timing horizons are explicit model restrictions. An
infeasible finite-horizon scheduling model is not a claim of infeasibility with
unbounded time. Use different overhead/capacity values to change which pipeline
alternatives win, and nondivisible output counts to exercise tails.

Each case gets one excluded warmup, followed by five measured repetitions by
default. `--warmups 0`, `--repeats`, and `--sample-work` are configurable. Bounds
are sampled at deterministic solver-work intervals with wall timestamps.
The default runs every saved case. An explicit `--stop-after-incomplete N`
can censor subsequent cases after N consecutive incomplete cases in a family;
those cases are saved as unrun in `skipped.json`. Use 0 to disable censoring.

The supervisor runs one child process at a time. It enforces a wall watchdog
(solver budget plus one second for setup), samples process RSS/CPU time through
`ps`, and kills a child exceeding the observed memory envelope. Cooperative
solver memory budgets separately bound retained search state. RSS sampling can
miss brief allocation peaks; missing observations remain null, never zero.
No large competing benchmark is launched concurrently by this harness.

## Reading the results

`*.run.json` retains model identity, policy, build/machine/source revision,
settings, first-witness time, eventual-winner time (completed optimal runs only),
proof time, solver counters, bounds and observed process memory/CPU use. Search
initialization/validation, solve and process-overhead times are separate.
`summary.json`, `summary.csv` and `report.md` contain median/range of completed
runs plus incomplete/error counts. Completion fractions use **all** measured
runs at 10 ms, 100 ms, 1 s, 10 s and 60 s. A timeout does not become a 60-second
completed proof. Per-run bound curves remain available for plotting.

Compare work counts at fixed coupling before fitting wall-clock scaling. A raw
product of local domain sizes is only a rectangular upper estimate and can
include inactive or illegal assignments. Neither that product nor a successful
small case predicts practical compilation time. Synthetic success must be
followed by models derived from representative actual compiler problems.

## Pinned CP-SAT reference

In a separate Python environment install `reference/requirements.txt`, which
pins OR-Tools 9.14.6206, then use `verify --reference cp-sat`. The adapter uses one
worker, fixed seed, and zero absolute/relative optimality gaps. Only `OPTIMAL`
and `INFEASIBLE` count as reference proofs. Other statuses are censored. Returned
integer witnesses and costs are checked by both the independent Rust evaluator
and the original model. Declared missing coverage conservatively leaves this
reference incomplete; it does not disprove a solver optimum that excludes the
affected region. Unsupported encodings or CP-SAT integer limits
are explicit errors, not silently omitted constraints. This optional dependency
is not installed by building the Rust solver and is never an execution fallback.

The local pilot evidence in [`evidence/cp-sat-corrections.json`](evidence/cp-sat-corrections.json)
covers the finite `packing-gap` (optimum 9) and `repeated-coupled` (optimum 8)
fixtures. It does not claim that every scheduling or pipeline family has a faithful
CP-SAT encoding, or that the production solver completes those families within a
useful budget.

The CLI CP-SAT comparison requires the pinned interpreter explicitly when OR-Tools
is installed outside PATH:

```sh
SOLVER_LAB_PYTHON=/tmp/magnitude-solver-reference-venv/bin/python \
  inference-v4/target/release/magnitude-solver-lab verify --input /tmp/solver-lab-corrections2 \
  --reference cp-sat --seconds-per-case 5 --out /tmp/magnitude-solver-evidence/cp-sat-verification.json
```

The successful two-fixture output is retained in
[`evidence/cp-sat-cli-verification.json`](evidence/cp-sat-cli-verification.json).

After installing the pinned requirements, run the independent encoding's
adversarial scheduling checks with that interpreter:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/magnitude-solver-reference-venv/bin/python inference-v4/solver/lab/reference/test_cp_sat.py
```

They cover inactive negative timing/demand fields, both-endpoint precedence
activation, empty intervals, half-open capacity release and guarded completion.

## Selectable neighborhood-search experiment

`search-study` compares saved, identical `Model` values through the shared solver
interface. `exact` retains its global proof semantics. The approximate methods
share initialization, scoped repair, propagation, validation and work accounting:

| Method | Neighborhood | Acceptance / restarts |
| --- | --- | --- |
| `greedy` | One seed variable plus typed dependency release | Improving moves, population 1, periodic restarts |
| `anneal` | One seed variable plus typed dependency release | Uphill exploration, population 1, periodic restarts |
| `joint` | Connected group plus dependency release | Improving moves, population 1, no restart |
| `lns` | Connected group plus dependency release | Uphill exploration, population and restarts |
| `random` | Independent complete samples with typed arithmetic/guard consequences | Keep best valid candidate |

The random sampler has its own simple constructor; it is a sampling control, not
an isolated acceptance-rule ablation. It is not uniform over legal assignments.
Every sampler improvement passes both the independent interpreter and original
model validator. Single-seed neighborhoods can release more than one variable
through dependencies: read the released-variable diagnostics before attributing
a result to one-coordinate moves. `joint` versus `lns` combines exploration,
population and restart changes; it is not a single-feature ablation.

`kernel-contraction` keeps tile geometry, staging and layout as distinct variables.
A typed product relates tile/staging to live storage; a single common capacity
couples regions. Its transfer, arithmetic, tail and conversion service costs
come from small local tables, never one index over complete kernel plans.
`kernel-shared` exposes sharing and retention globally and tile/fusion/layout per
consumer. Costs include one versus repeated preparation, materialization,
conversion and reload work. These two families have an additive modeled service
objective under a workspace capacity; **they are not parallel makespan models**.
Their independent capacity-DP oracles are audited against whole-assignment
interpretation on tiny instances.

`kernel-schedule` separately tests variable-duration instruction modes that trade
latency against resource demand. Starts and ends remain explicit finite variables.
An independent mode/start enumeration supplies OPT and, offline, the selected
implementation's optimum schedule f*(x). The returned cost U separates schedule
loss U/f*(x) from implementation loss f*(x)/OPT. It is a small bounded scheduler
fixture, not evidence for million-iteration coupled schedules.

```sh
cargo test --manifest-path inference-v4/solver/lab/Cargo.toml
cargo build --release --manifest-path inference-v4/solver/lab/Cargo.toml
inference-v4/target/release/magnitude-solver-lab generate --suite lns-development --seeds 0..9 --out /tmp/solver-lns/dev-inputs
inference-v4/target/release/magnitude-solver-lab search-study --input /tmp/solver-lns/dev-inputs --out /tmp/solver-lns/dev --search-seeds 0,1,2,3,4 --milliseconds 1000
```

Freeze tuning after development, then generate `lns-held-out` with seeds
`100..119`. This suite also changes capacities, overhead, extents and schedule
resource regimes. Save the actual settings in every result; changing defaults
requires a new cohort. `lns-scaling` varies region count, tile domain, shared
capacity and independent additive repetition separately. The latter explicitly
makes no claim about coupled parallel repetition or state-carrying loops.

`search-study` uses sequential child processes with a readiness marker, watchdog and RSS sampling. OS loader startup has a separate 120-second grace and is classified separately from algorithm timeouts.
Every child regenerates its saved fixture, verifies model equality, constructs
its search and independently checks every new incumbent within the timed budget.
Reported points include this reconstruction/initialization/validation cost;
process launch and independently bounded reference solving are excluded. Samples
are timestamped after witness checking: a result finishing after a checkpoint is
never credited to the earlier checkpoint. Reference timeouts remain unresolved.

`cohort.json` records the machine, Rust version, source revision/dirty state,
command and frozen settings. Instance JSON, oracle JSON and each full witness,
quality curve and cumulative solver stats are retained. `summary.json` separates
family and exact parameter regime and reports checkpoint feasibility, regret
p50/p95/max, severe outliers and time/work to 5% regret. Target-time medians are
conditional on success: compare success rates before claiming a speedup.
The study summaries select sorted index `ceil((n - 1) * p)` for percentile `p`;
their median fields therefore select the upper middle observation for even
sample counts. The separate paired-analysis script uses arithmetic medians.
Random-sampler work counts mean complete attempts, not solver work units.
Use `study-report --input DIR/runs --out DIR` to regenerate a report.

The initial experiment can fail quality, relevance or speedup gates. It must not
change production compiler selection merely because valid heuristic incumbents
are available. Source-derived unresolved implementation models and compact
stateful coupled repetition remain separately identified evidence requirements.

The [2026-09-18 evidence report](results/2026-09-18-neighborhood/README.md) retains
2,508 development, held-out, scaling and width-only runs. It supports coordinated
improving moves on these synthetic models while rejecting the current full
exploration configuration as a production default.

## Qwen source-family qualification

The optional `qwen-source` binary uses the engine's authored Qwen3.5 dense,
attention and recurrent compositions and standard kernels. It measures source
family construction and constraint export separately. It deliberately reports
`quality_eligible: false`: these models do not yet include complete conditional
backend schedules and costs. Component probes are not a replacement for the
runtime's enclosing-composition selection; `*-entry` cases exercise that larger
source boundary too.

```sh
cargo build --release --manifest-path inference-v4/Cargo.toml \
  -p magnitude-solver-lab --features qwen-source --bin qwen-source
hf download mlx-community/Qwen3.5-4B-4bit config.json model.safetensors.index.json \
  --revision 0e7ffd5c629ef7719d4cbc04069232580bfa9d9c \
  --local-dir inference-v4/solver/lab/results/qwen/config
python3 inference-v4/solver/lab/scripts/qwen_source.py \
  --binary inference-v4/target/release/qwen-source \
  --config-dir inference-v4/solver/lab/results/qwen/config \
  --out inference-v4/solver/lab/results/qwen/runs --repeats 3
```

The runner validates model geometry and quantized recurrent projection metadata,
records the binary/configuration hashes, and bounds each case to ten seconds
following its readiness marker (loader startup has a separate 60-second cap).
Raw results, configuration downloads and saved executables stay ignored.
