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

The upstream numerical implementation remains authoritative in this pass-through
assembly. [Owned kernel construction](../../kernels.md) does not require translating
it into an owned execution IR.

## Assembly

```text
MODEL:EXECUTOR:MAG:STANDARD
├── Upstream language model · MODEL:FORWARD:VLM:STANDARD
└── Native cache adapter · STATE:CHECKPOINTS:MAG:NATIVE
```

## Component definitions

Each type below defines its behavior and mathematical assumptions. Executable bindings
are owned by the [performance catalog](../../performance.md#ownership-and-component-records).
Parameters inherit the [origin/platform rules](../../performance.md#dimensions-and-parameter-binding).
`JOIN` and `L` use the [resource algebra](../../performance/derivations/resources.md#evaluation-algebra);
[neural regions](../../performance/derivations/neural.md#named-region-bindings) supply the shared terms.
Implementation estimates use selected execution regions and matched local/parent
observations under [execution estimation](../../performance/derivations/resources.md#execution-estimation).
References and tests describe controls; they do not assert current performance qualification.

### `MODEL:EXECUTOR`

**Contract.** Supply inputs/positions, invoke the bound language forward and publish valid outputs/state
while preserving allocation/completion obligations.

**Parameters.** Architecture: selected upstream forward graph and supported native cache/numerical contract.
Workload: `b,q`, histories, requests, adaptation layout and residency.

**Composition.** `D_EXECUTOR=JOIN(D_FORWARD,required_input/state/output_adaptation)` using [resource
algebra](../../performance/derivations/resources.md#evaluation-algebra), [forward](#modelforward) and [native
checkpoints](#statecheckpoints). Add only newly compulsory boundary information; no fixed
Python wrapper floor. Actual adapter work belongs to its execution estimate.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:EXECUTOR/EXEC` | Elapsed seconds for `u=bq` consumed inputs through outputs/state and required completion. | `L(D_EXECUTOR)`. |

**Implementations and controls.**

#### `MODEL:EXECUTOR:MAG:STANDARD`

- **Implementation:** Tokens and logical starting state become requested logits and advanced state. Own position
  binding, compatible batch assembly, completion roots and resource lifetime around the two
  runtime children above. No added whole-model compilation; unrequested logits remain
  unevaluated where their computation is exclusive to those outputs.
- **Reference / validation:** Invoke a separately loaded stock MLX-VLM model with the same artifact, tokens, positions and
  state. Compare logits and subsequent state behavior for single requests, batch changes and
  restored prefixes.


### `MODEL:LOADING`

**Contract.** Resolve configuration/modules, validate supported text tensors, materialize weights and own
their budgeted lifetime; failures release acquired resources.

**Parameters.** Architecture: required final weight encoding and unique tensor identities. Workload: initial
artifact/page-cache residency, conversion requirements and observation interval; platform
supplies storage/memory/conversion upper capacities.

**Composition.** Bind [loading](../../performance/derivations/state.md#loading): pipeline necessary reads/conversion, union tied weights
and omit unproved temporary-memory floors. Startup latency and peak footprint can trade off
through concurrent staging/conversion. Cold/warm loading are operating points, not separate
dimensions.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:LOADING/LAT` | Seconds from the specified initial artifact/residency state to usable materialized text weights. | `L_load`; efficiency `100*L_load/T_load`. |
| `MODEL:LOADING/MEM` | Peak live bytes attributable to loading and resulting resident weights over that same interval, including staging/conversion and counting shared backing once. | `M_load_min`; efficiency `100*M_load_min/M_peak`. |

**Implementations and controls.**

#### `MODEL:LOADING:MAG:RESIDENT`

- **Implementation:** Resolve upstream configuration and language module, validate tensor layout, materialize
  supported text weights and own their budgeted lifetime. Peer modality weights are excluded.
  Loading failures release resources.
- **Reference / validation:** Compare tensor values, names, encodings and model arguments with direct upstream loading.
  Exercise unsupported layouts and partial failures.

### `MODEL:FORWARD`

**Contract.** Execute the selected upstream language model’s mathematical graph, returning requested
logits/features and advancing its declared cache state.

**Parameters.** Architecture: explicit upstream operator graph, tensor geometry/encoding, sharing and state
obligations. Workload: `b,q`, histories, requested outputs and residency. A library name alone
does not instantiate these parameters.

**Composition.** `D_FORWARD=JOIN(actual_operator_graph,required_external_state/outputs)` using
[neural](../../performance/derivations/neural.md) and [resource](../../performance/derivations/resources.md#evaluation-algebra) derivations.
[Qwen](qwen35.md#modelqwen35) and [Gemma](gemma4.md#modelgemma4) bind matching equations;
other upstream architectures supply their graph explicitly. Unbound graphs stay
uninstantiated, and library timings never fill the theoretical denominator.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `MODEL:FORWARD/EXEC` | Elapsed seconds for `u=bq` consumed inputs through requested neural outputs/state. | `L(D_FORWARD)`. |

**Implementations and controls.**

#### `MODEL:FORWARD:VLM:STANDARD`

- **Implementation:** The pinned upstream standalone language model consumes tokens/native caches and produces
  logits with updated caches. Its blocks, kernels and numerical conventions remain
  upstream-owned.
- **Reference / validation:** Direct invocation is the reference for our integration. It cannot independently validate its
  own equations: use an independent model implementation or mathematical block oracle when those
  equations are in question.


### `STATE:CHECKPOINTS`

**Contract.** Adapt supported upstream caches to reserve, begin, advance, checkpoint and restore operations
with row isolation, budget checks and complete state lifetimes.

**Parameters.** Architecture: actual cache types/geometries/encoding and allowed reconstruction. Workload:
retained/visible histories, checkpoint obligations, advanced/accepted positions, memory budget
and observation boundary.

**Composition.** Use the [required live union](../../performance/derivations/state.md#required-live-union) over native cache information
and [restoration cases](../../performance/derivations/state.md#restoration-cases). More retained images may speed
restore while increasing footprint. Window visibility during a wide advance differs from
retained history; replacement peaks and creation/advance work remain constraints and enclosing
costs.

**Dimensions.**

| ID | Metric and boundary | Theoretical bound |
|---|---|---|
| `STATE:CHECKPOINTS/MEM` | Retained physical bytes for live state and required restorable checkpoints at the specified lifecycle boundary; shared backing once. | Required materialized union `M_min`; efficiency `100*M_min/M_retained`. |
| `STATE:CHECKPOINTS/RESTORE` | Seconds to accepted-checkpoint readiness from the specified advanced state, including deferred repair before next use. | `L_restore(initial,accepted,obligations,budget)`; efficiency `100*L/T`. |

**Implementations and controls.**

#### `STATE:CHECKPOINTS:MAG:NATIVE`

- **Implementation:** Wrap supported upstream cache objects with reserve, begin, advance, checkpoint and restore
  behavior. Preserve logical positions and state isolation; account for replacement peaks before
  execution. Shared by generic targets and compatible attached heads, with model-specific
  capacity geometry.
- **Reference / validation:** Compare independently advanced upstream caches and exact logical checkpoint contents. Exercise
  window crossings, multi-input extensions, rejected suffixes, batch changes and budget failure
  before mutation.


## Qualification

Current scores follow the [assessment rules](../../performance.md#stable-compositions-and-evidence).

These IDs describe existing responsibilities; they do not assert universal upstream
parity. The September 6 native Gemma comparison established matching logits for the
recorded 1K/16K/32K cases after window-accounting correction. It did not establish
rotating-cache batching, broad artifact support or a speedup. New source revisions
and model/cache combinations require their own evidence keyed to the same IDs.

Evidence: `sessions/26-09-06/evidence/cycle-006/comparison.json` and
`qualification.manifest.json` in that directory, relative to the monorepo root.
