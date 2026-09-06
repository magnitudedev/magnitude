# Component identification and assembly

**Stable identifiers name implementations. Assemblies connect those implementations
into models and engines; evidence establishes what each implementation achieves.**

This vocabulary applies across the engine. [Model composability](models/composability.md)
defines computational contracts and substitution; [model optimization](models/optimization.md)
defines performance reasoning and qualification.

## Identifier grammar

```text
FAMILY:COMPONENT:SOURCE:VARIANT
```

Use uppercase names. Colons separate fields; dots express hierarchy inside a
component address. Use readable words, with underscores for multiword names.
Source codes are the deliberate abbreviations.

| Field | Meaning | Examples |
|---|---|---|
| Family | The subsystem containing the component | `MODEL`, `SCHEDULING`, `KV` |
| Component | Its responsibility within that family | `QWEN35.ATTENTION`, `ADMISSION`, `APPEND` |
| Source | Who supplies the implementation or defines its composition | `MLX`, `LM`, `VLM`, `MAG` |
| Variant | The implementation's distinguishing method | `FUSED`, `PAGED`, `FIFO` |

```text
MODEL:QWEN35.ATTENTION:VLM:STANDARD
MODEL:QWEN35.ATTENTION:MAG:FUSED
MODEL:ATTENTION:MLX:DENSE
SCHEDULING:ADMISSION:MAG:FIFO
KV:APPEND:MAG:PAGED
```

The component address follows its family's concepts; there is no universal scope
field. Within `MODEL`, an architecture qualifier such as `QWEN35` or `GEMMA4`
identifies family-specific computation. An unqualified address such as `ATTENTION`
identifies shared model computation, subject to its declared capabilities. This
distinction does not impose a generic/specific classification on other families.

## Vocabulary and uniqueness

| Family | Responsibility |
|---|---|
| `ENGINE` | Complete engine composition |
| `SCHEDULING` | Admission, service selection and allowances |
| `BATCHING` | Assembly of compatible ready work |
| `GENERATION` | Token advancement, drafting, verification and acceptance |
| `EXECUTION` | Device submission, completion and resource lifetime |
| `MEMORY` | Capacity accounting and reservations |
| `CACHE` | Reusable prefix indexing and retention |
| `STATE` | Logical model state and checkpoint composition |
| `KV` | Physical attention-history storage and operations |
| `MODEL` | Neural architectures and computational blocks |

Source codes are `MLX` for MLX, `LM` for MLX-LM, `VLM` for MLX-VLM, and `MAG`
for Magnitude. Additional sources need explicit codes rather than a catch-all.

Each `FAMILY:COMPONENT` address names one contract. Each full ID names one
implementation of that contract. Its owning design document defines the ID and
its meaning; other documents reference that definition. Reuse existing names
rather than inventing synonyms. New families, addresses and variants require
definitions at the same time as their introduction.

Variants describe methods, not rankings or benchmark results. `STANDARD` denotes
the source's conventional implementation where no further distinction is needed;
it does not mean correct by definition or preferred. Avoid opaque numbers and
labels such as `FAST`, `BEST` or `V2`.

## Identity and provenance

An ID identifies an implementation, not a loaded instance. Many layers can use
the same implementation with different weights. Configuration, artifact identity,
source revisions, dependency versions, hardware and qualification evidence attach
to the ID rather than becoming part of it.

File moves and ordinary improvements preserve identity. Separately selectable
methods get distinct variant IDs. An incompatible contract needs a distinct
component address; an old ID must not silently acquire a different meaning.
Measurements always identify the measured revision as well as the component ID.

Source attribution follows composition ownership recursively:

- A Magnitude composition of upstream and owned pieces is `MAG`.
- An MLX-VLM composition of MLX operations is `VLM`.
- A pass-through preserves the upstream component's identity. Any wrapper with
  its own substantive behavior is described separately.

Children retain their sources. A parent does not concatenate their source codes
or become `MIXED`; the assembly records that provenance. This is ownership
precedence at each composition boundary, not a ranking that lets any `MAG` child
automatically relabel its enclosing upstream component.

Implementation technology is a separate property. An owned Metal kernel has
source `MAG`, technology `MTL`, and can execute through runtime `MLX`. An upstream
MLX primitive remains source `MLX` even when its backend uses Metal. `MTL` is not
a source code, and a composed block need not have one technology for all children.

## Assemblies and component descriptions

An assembly names selected implementations and their relationships. Dotted
addresses identify responsibilities; nesting shows actual composition. Shared
state, weights or execution resources are explicit connections, so an assembly
is a graph even when displayed as a tree.

Every component in an architecture or engine assembly resolves to one authoritative
definition. The same ID connects that definition to implementation bindings,
independent tests, benchmark subjects and result records. Definitions contain:

| Property | Required description |
|---|---|
| Contract | Computation or policy, inputs/outputs, state effects and lifetime guarantees |
| Support | Relevant shapes, precision, storage capabilities and configuration constraints |
| Composition | Constituent IDs, their roles and shared dependencies |
| Implementation | Source, method and relevant technology/runtime distinctions |
| Comparison | Available reference implementations or independent oracles, the boundary each validates, and conditions required for a fair comparison |
| Performance properties | Relevant latency, throughput, traffic, memory or service properties and how they scale with the supported workload |
| Performance model | Derivation from required work and resource capacity, composition of child costs, or reference-based targets; assumptions, remaining headroom and unknowns |
| Validation | How to exercise the component independently with matched inputs/state and observe outputs, state effects and performance |
| Qualification | Implemented versus proposed status, supported claims, comparison results and source/configuration/evidence identity |

This applies to primitives, composed blocks and complete engines. Each definition
explains why its references are appropriate and which properties they establish.
A reference can be an upstream implementation, an independently qualified internal
implementation or a mathematical oracle. If none is available, record the gap and
the independent validation needed; naming a reference is not proof of equivalence.

Every component needs an explicit performance model, even before a numerical
ceiling is established. Distinguish demonstrated reference performance from a
derived bound. State the quantities, assumptions and missing measurements needed
to evaluate the model; "not benchmarked" alone is insufficient. Policy components
can have service, interference and overhead bounds rather than a token-rate ceiling.

A composite model names its child IDs and explains serial dependencies, feasible
overlap, shared resources and additional composition costs. It cannot simply sum
isolated timings or multiply speedups. A primitive derives its model from its own
algorithm and resource demands. Both may use independent references to challenge
the derivation. Observed implementation costs do not become permanent limits.

Assembly trees reference these definitions rather than duplicating their claims.
Test and benchmark records retain the component ID, actual child selection,
revision, workload and reference identity, so a result can be traced to both a
specific implementation and its enclosing composition. A testable boundary is an
observable contract; it need not be a separate kernel or production dispatch.

Matching component addresses makes implementations candidates for comparison and
substitution; supported layouts and capabilities must still match or be explicitly
adapted. A reference relationship is not a production dependency.

IDs do not require a runtime registry, object or dispatch for every component.
Model blocks can fuse and compile across their identified boundaries. Diagnostic
boundaries must not force production intermediates or synchronization.

## Worked assembly

The following is an illustrative composition using the vocabulary above. It
shows how a compiled Qwen target could fit into an engine; it is not a declaration
that every named optimization is implemented or qualified.

```text
ENGINE:INFERENCE:MAG:STANDARD
├── SCHEDULING:SERVICE:MAG:TIME_SHARING
│   ├── SCHEDULING:ADMISSION:MAG:FIFO
│   └── SCHEDULING:PREFILL:MAG:CHUNKED
├── BATCHING:ASSEMBLY:MAG:READY_COMPATIBLE
├── MEMORY:ACCOUNTING:MAG:RESERVATIONS
├── CACHE:PREFIX:MAG:CHECKPOINTS
├── EXECUTION:DEVICE:MAG:ASYNC
├── STATE:QWEN35:MAG:HYBRID
│   ├── KV:STORE:MAG:PAGED
│   │   ├── KV:APPEND:MAG:CONTIGUOUS_RUNS
│   │   └── KV:BRANCH:MAG:COPY_ON_WRITE
│   └── STATE:RECURRENT:MAG:CHECKPOINTED
└── GENERATION:SPECULATION:MAG:TARGET_MATCHING
    ├── target → MODEL:QWEN35:MAG:COMPILED
    └── drafter → MODEL:QWEN35.MTP:MAG:CONDITIONED
```

The target expands into model components. Its layer definition selects attention
or recurrence according to the architecture; the shared implementations retain
the same IDs across layers and models.

```text
MODEL:QWEN35:MAG:COMPILED
├── MODEL:EMBEDDING:MLX:QUANTIZED
├── MODEL:QWEN35.LAYER:MAG:COMPILED                  repeated per configuration
│   ├── MODEL:NORMALIZATION:MLX:RMS
│   ├── mixer: one of
│   │   ├── MODEL:QWEN35.ATTENTION:MAG:FUSED
│   │   │   ├── MODEL:PROJECTION:MAG:PACKED
│   │   │   │   └── MODEL:PROJECTION:MLX:QUANTIZED
│   │   │   ├── MODEL:QWEN35.ATTENTION.PREPARATION:MAG:FUSED
│   │   │   ├── MODEL:ATTENTION:MAG:PAGED
│   │   │   └── MODEL:QWEN35.ATTENTION.OUTPUT:MAG:GATED
│   │   └── MODEL:QWEN35.RECURRENCE:MAG:COMPILED
│   │       ├── MODEL:PROJECTION:MAG:PACKED
│   │       ├── MODEL:QWEN35.RECURRENCE.PREPARATION:MAG:FUSED
│   │       ├── MODEL:GATED_DELTA:LM:STANDARD
│   │       └── MODEL:QWEN35.RECURRENCE.OUTPUT:MAG:GATED
│   ├── MODEL:RESIDUAL_NORMALIZATION:MAG:FUSED
│   ├── MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED
│   │   ├── MODEL:QWEN35.ROUTER:MAG:FUSED
│   │   ├── MODEL:EXPERTS:MAG:FUSED
│   │   ├── MODEL:FEEDFORWARD:MAG:SWIGLU
│   │   └── MODEL:QWEN35.EXPERT_COMBINATION:MAG:FUSED
│   └── MODEL:RESIDUAL:MLX:ADD
├── MODEL:NORMALIZATION:MLX:RMS
└── MODEL:PROJECTION:MLX:QUANTIZED
```

The target uses the engine's hybrid state and execution owner; nesting does not
create private copies of those resources. A paged attention implementation declares
its required KV view without taking ownership of prefix policy or admission.

For comparison, `MODEL:QWEN35.ATTENTION:VLM:STANDARD` can control the complete
attention block, while `MODEL:ATTENTION:MLX:DENSE` can control prepared attention
computation with equivalent logical KV. These references validate different
boundaries; neither alone qualifies the full engine.
