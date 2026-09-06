# Component ceiling catalog

This catalog binds every contract currently identified in the model docs to an
optimistic [MLX ceiling](../performance.md). Source and variant select the measured
implementation; they do not select a more favorable theoretical denominator.

All entries currently have **symbolic bounds only**. No runtime capacities, measured
rates or efficiency percentages are assigned here. Each row inherits the linked
derivation's assumptions and relaxations. Geometry comes from the bound artifact
and workload, never a model-family name alone.

## Dimension definitions

Every row below declares its complete set of dimensions. `EXEC`
means execution efficiency: elapsed seconds for the specified useful operation,
efficiency `L / T` under the
[time derivation](../performance.md#meaning-of-the-percentage). The operation and
its geometry are operating-point coordinates, including query width and context;
prefill/decode are not extra dimensions. Each row explicitly binds this definition
to its full dimension ID, including single-dimension contracts. A zero/unresolved
bound has no percentage.

Only the following contracts currently need multiple dimensions. These definitions
are shared by all implementations of each contract; each evaluated sample supplies
the specified workload and boundaries.

| Dimension ID | Metric and boundary | Bound and efficiency |
|---|---|---|
| `MODEL:LOADING/LAT` | Seconds from a specified initial artifact/residency state to usable, materialized text weights | [Loading time](derivations.md#state-loading-and-execution): `L_load / T_load` |
| `MODEL:LOADING/MEM` | Peak live bytes attributable to loading and its resulting resident weights over that same interval; include staging/conversion allocations, count shared physical allocations once | [Loading footprint](derivations.md#footprint-and-restoration): `M_load_min / M_load_peak` |
| `STATE:CHECKPOINTS/MEM` | Retained physical bytes for the live logical state and required restorable checkpoints at a specified lifecycle boundary; count shared backing once | [State footprint](derivations.md#footprint-and-restoration): `M_state_min / M_state_retained`, instantiated for the selected native cache geometry |
| `STATE:CHECKPOINTS/RESTORE` | Seconds to make a specified accepted checkpoint usable from a specified advanced state, including deferred repair required before its next use | [Restoration](derivations.md#footprint-and-restoration): `L_restore / T_restore`, with checkpoint obligations and memory budget fixed |
| `STATE:QWEN35/MEM` | Retained physical bytes for attention history, convolution/recurrent state and required restorable checkpoints at a specified lifecycle boundary; count shared backing once | [State footprint](derivations.md#footprint-and-restoration): `M_state_min / M_state_retained`, instantiated for Qwen's hybrid state geometry |
| `STATE:QWEN35/RESTORE` | Seconds to restore both attention and recurrent state to a specified accepted boundary, including deferred repair required before its next use | [Restoration](derivations.md#footprint-and-restoration): `L_restore / T_restore`, including any unavoidable recurrent work under the fixed memory budget |

Loading can reduce latency by staging/converting more weights concurrently, increasing
peak memory. State can reduce restoration time by retaining more snapshots, increasing
retained memory. These outcomes warrant separate dimensions; neither compensates
silently for the other. State transient peaks and advance/checkpoint creation costs
remain required diagnostics and budget/end-to-end constraints. They are not hidden by
its retained-memory score. Loading's `MEM` explicitly means peak footprint; state's
`MEM` means retained footprint. References always use the full dimension ID.

## Shared model contracts

| Contract | Dimension IDs | Derivation and parameter binding |
|---|---|---|
| `MODEL:EMBEDDING` | `MODEL:EMBEDDING/EXEC` | [Embedding](derivations.md#embedding-and-elementwise-regions): unique token rows, width, actual row encoding and boundary residency |
| `MODEL:EXPERTS` | `MODEL:EXPERTS/EXEC` | [Projections/experts](derivations.md#projections-and-experts): gate/up/down shapes, encoding, assignments and unique experts; activation and required per-expert outputs |
| `MODEL:ATTENTION` | `MODEL:ATTENTION/EXEC` | [Attention](derivations.md#attention): per-row old length, query width, head geometry, dtype/window and boundary-visible K/V; [memory boundary](derivations.md#notation-and-memory-boundary) handles dense/paged adaptation |
| `MODEL:GATED_DELTA` | `MODEL:GATED_DELTA/EXEC` | [Recurrence](derivations.md#gated-delta-recurrence): state geometry, prepared inputs, output requirements and query count; no compulsory per-token state spill |

These bind the [shared implementations](../models/composability.md#shared-component-definitions),
including upstream controls. Unproved layout overhead is excluded from the common
ceiling; it remains visible when diagnosing the actual implementation.

## Generic upstream contracts

| Contract | Dimension IDs | Derivation and parameter binding |
|---|---|---|
| `MODEL:EXECUTOR` | `MODEL:EXECUTOR/EXEC` | [Composition](derivations.md#parent-and-engine-composition): actual upstream graph plus required state/output boundaries; no invented adapter floor |
| `MODEL:LOADING` | `MODEL:LOADING/LAT`, `MODEL:LOADING/MEM` | [Loading](derivations.md#state-loading-and-execution): missing text-weight bytes, resident representation and conversion requirements; startup only |
| `MODEL:FORWARD` | `MODEL:FORWARD/EXEC` | [Composition](derivations.md#parent-and-engine-composition): instantiate actual architecture blocks; upstream library rate is not a ceiling |
| `STATE:CHECKPOINTS` | `STATE:CHECKPOINTS/MEM`, `STATE:CHECKPOINTS/RESTORE` | [State](derivations.md#state-loading-and-execution): logical retained/visible state, immutable sharing, required restoration and memory budget |

See the [generic assembly](../models/architectures/generic-mlx-vlm.md).

## Qwen contracts

| Contract | Dimension IDs | Derivation and parameter binding |
|---|---|---|
| `MODEL:QWEN35` | `MODEL:QWEN35/EXEC` | [Composition](derivations.md#parent-and-engine-composition): actual attention/recurrent layer sequence, dense/routed branches, embedding and requested head; remove fusible boundary traffic |
| `MODEL:QWEN35.ATTENTION` | `MODEL:QWEN35.ATTENTION/EXEC` | [Projections](derivations.md#projections-and-experts) + [elementwise](derivations.md#embedding-and-elementwise-regions) + [attention](derivations.md#attention) + new persistent KV; include gate and output projection, deduplicate prepared K/V |
| `MODEL:QWEN35.RECURRENCE` | `MODEL:QWEN35.RECURRENCE/EXEC` | Input/output [projections](derivations.md#projections-and-experts), convolution/gates/norms and [recurrence](derivations.md#gated-delta-recurrence); fuse preparation, retain only required final/checkpoint state |
| `MODEL:QWEN35.FEEDFORWARD` | `MODEL:QWEN35.FEEDFORWARD/EXEC` | [MLP/MoE](derivations.md#projections-and-experts): configure dense or routed/shared branches, router dimensions and required selection semantics; permit packed projections and fused reduction |
| `MODEL:QWEN35.READOUT` | `MODEL:QWEN35.READOUT/EXEC` | [Projection](derivations.md#projections-and-experts) with vocabulary width and requested rows, plus final norm; preserve tied-weight sharing where configured |
| `STATE:QWEN35` | `STATE:QWEN35/MEM`, `STATE:QWEN35/RESTORE` | [State](derivations.md#state-loading-and-execution): unique attention histories plus fixed-size recurrent state and restoration obligations; do not double-charge writes inside mixers |
| `MODEL:QWEN35.MTP` | `MODEL:QWEN35.MTP/EXEC` | [Composition](derivations.md#parent-and-engine-composition): conditioning projections/norms, head layers, borrowed vocabulary and native state; round efficiency belongs to generation, not this head alone |

See the [Qwen assembly](../models/architectures/qwen35.md). A static worksheet under
`sessions/26-09-06/evidence/ceiling-derivations/` records the pinned Qwen3.6 Q4 artifact's
configuration/header geometry without loading tensors or executing MLX. Its byte
counts are logical stored/accessed payloads, not asserted DRAM traffic or timings.

## Gemma contracts

| Contract | Dimension IDs | Derivation and parameter binding |
|---|---|---|
| `MODEL:GEMMA4` | `MODEL:GEMMA4/EXEC` | [Composition](derivations.md#parent-and-engine-composition): configured branches, local/global attention, producer sharing, layer inputs/scales and requested head |
| `MODEL:GEMMA4.ATTENTION` | `MODEL:GEMMA4.ATTENTION/EXEC` | Q/output [projections](derivations.md#projections-and-experts), normalization/rotary and [attention](derivations.md#attention); KV production only for writers, optimistic reuse across readers |
| `MODEL:GEMMA4.KV` | `MODEL:GEMMA4.KV/EXEC` | K and optional V [projections](derivations.md#projections-and-experts), distinct branch transforms and [append](derivations.md#state-loading-and-execution); K=V removes only the raw duplicate projection |
| `MODEL:GEMMA4.FEEDFORWARD` | `MODEL:GEMMA4.FEEDFORWARD/EXEC` | [Composition](derivations.md#parent-and-engine-composition): all enabled dense/routed branches and normalization order; permit fusion without deleting required branches |
| `MODEL:GEMMA4.MLP` | `MODEL:GEMMA4.MLP/EXEC` | [MLP](derivations.md#projections-and-experts): actual gate/up/down widths and encoding; GeGLU intermediates need not leave the parent |
| `MODEL:GEMMA4.EXPERT_BRANCH` | `MODEL:GEMMA4.EXPERT_BRANCH/EXEC` | [MoE](derivations.md#projections-and-experts): router, selected-score softmax/scales, expert assignments and output norm; preserve Gemma-specific equations |
| `MODEL:GEMMA4.INPUTS` | `MODEL:GEMMA4.INPUTS/EXEC` | [Embedding/projections](derivations.md#embedding-and-elementwise-regions): prepare once, apply per layer; bound workspace separately from repeated traffic |
| `MODEL:GEMMA4.READOUT` | `MODEL:GEMMA4.READOUT/EXEC` | Vocabulary [projection](derivations.md#projections-and-experts), final norm and configured soft cap; only required output rows, fusible intermediates |

See the [Gemma assembly](../models/architectures/gemma4.md). Its artifact geometry and
platform profile remain parameters; Qwen's instantiated quantities do not transfer.

## Engine scope

The model catalog does not yet instantiate a particular engine workload. The
[engine derivation](derivations.md#parent-and-engine-composition) supplies its rule:
state the service objective and constraints, compose mathematical model demand,
then add only provably unavoidable coordination costs. The illustrative engine IDs
in the identification document are not measured entries in this catalog.

## Updating a binding

A formula revision changes the assessment, not the historical observation. Record
full dimension ID, formula content revision, complete parameter binding, profile
identity, implementation/child fingerprints and observation identity when evaluating
a row. Apply the [assessment reset rules](../performance.md#evidence-and-current-assessments)
to implementation changes and affected parent compositions.
Keep numerical evaluations outside these durable definitions. Missing parameters
stay explicit; references and current implementation timings cannot fill them by fiat.
