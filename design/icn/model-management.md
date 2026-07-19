---
applies_to:
  - inference/crates/icn-models/**
  - inference/crates/icn-contracts/src/inventory.rs
  - inference/crates/icn-reasoning/**
  - inference/crates/icn-hardware/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
---

# ICN model management

ICN maintains a durable inventory of locally usable models. Listing models is the single authoritative operation that reconciles the inventory and completes model inspection. Startup, filesystem changes, downloads, deletion, and model loading may update or invalidate known facts, but they do not independently start an inventory-wide inspection.

This design intentionally makes model listing an authoritative, potentially slower query:

> Reconcile the durable observation, complete every required inspection, and then return it.

It is not a snapshot of partially completed background work.

## Successful-list invariant

Every model returned with `Available` status has a complete, current assessment. ICN has established that:

1. The complete component set exists and has a stable content identity, including shards, projector, and MTP components where applicable.
2. The artifacts are valid GGUF files accepted by the pinned ICN runtime.
3. The exact effective chat template has been resolved with the same template-selection, fallback, BOS, EOS, and named-template behavior used when loading the model.
4. Every reasoning control supported by ICN has been conclusively classified as `Supported` or `Unsupported`.
5. The canonical product execution profile has been assessed against the current backend and hardware topology.
6. Hardware has been conclusively classified as `Fits` or `DoesNotFit`.
7. Any cached result matches every input that can affect the result.

Consequently, a successful list response must not expose `Pending`, `Assessing`, `NotAssessed`, or a generic `Unknown` for an available model. Those values describe orchestration progress or an incomplete implementation, not durable model properties.

Non-ready records, such as an in-progress or interrupted download, may still appear with their operation status. They are not `Available` models and are outside the completed-assessment invariant until their artifact set is ready.

## Authoritative reconciliation

The list-models handler calls one shared operation, referred to here as `ensure_model_inventory`:

1. Read and schema-validate the durable inventory index.
2. Enumerate configured model sources.
3. Reconcile additions, removals, content changes, and component-grouping changes.
4. Validate cached inspection and hardware evidence.
5. Inspect only new or stale candidates.
6. Atomically persist completed results.
7. Return one complete snapshot.

Concurrent list requests share the same in-flight ensure operation. They must not duplicate scans or native assessments.

Startup initializes storage, recovers interrupted operations, and may hydrate the durable index, but does not inspect every model. Download and acquisition completion directly records or invalidates the affected artifact. Deletion removes or invalidates the affected entry. Filesystem and runtime watchers only invalidate relevant cache entries. None of these paths launches inventory-wide enrichment.

## Reasoning discovery

Reasoning discovery describes the mechanical controls exposed by the effective chat template and
normalizes them to an ordered option list with a default. A model with no detected reasoning
behavior has the complete profile `none`, defaulting to `none`. Fixed reasoning has a non-disableable
enabled profile. The detailed normalization and family examples are defined in
[reasoning detection](./reasoning-detection.md).

Inspection must use the same effective template inputs as model execution. A missing template in GGUF metadata does not imply an unknown result: the pinned llama.cpp fallback behavior must be applied. Actual BOS and EOS tokens and any applicable named template variant must also be resolved rather than omitted.

Probe errors must not be converted to the `none` profile. `none` is valid only after inspection
successfully establishes absence of supported reasoning behavior. Likewise, a control that accepts
a single alternate effort level, a non-default toggle value, or another supported domain shape must
not be discarded merely because the current probe implementation fails to represent it. Such a case
is an inspector defect until the public contract deliberately excludes that control.

The discovery boundary is intentionally narrower than model intelligence. ICN can determine which supported controls a template exposes and how those controls affect rendering. It cannot infer from weights whether a model is good at reasoning, and it cannot promise to enumerate arbitrary values accepted by an unrestricted template program. The public capability contract must therefore describe the finite control vocabulary ICN supports. Within that declared vocabulary, discovery is expected to be total.

## Hardware discovery

Hardware assessment is made for a versioned canonical product execution profile. Given that profile, the complete result is:

```text
Hardware = Fits(profile, memory, recommendation)
         | DoesNotFit(profile, memory deficit, limiting resource, alternative)
```

`DoesNotFit` is a complete and successful result. It is not an assessment failure.

The profile fixes every execution input that can affect the result, including context length, sequence and batch sizing, KV types, acceleration and GPU-layer policy, projector or MTP selection, and capacity policy. The native planner reads model structure, enumerates devices, constructs the model and context plan, accounts for model, context, KV, compute, projector, and MTP memory, and evaluates the preferred and permitted fallback configurations.

Inventory assessment is advisory for the canonical profile. Loading a model still performs an exact safety assessment for the execution plan actually requested. The loader must not rely on a cached inventory assessment when those plans differ.

Native hardware assessment is currently serialized where required by process-global llama.cpp state. Other metadata and template work may use bounded concurrency. Implementation limits on concurrency do not weaken the completeness requirement.

## Exact failure taxonomy

The ordinary expectation is that every valid, stable model supported by the pinned runtime produces all required properties. Failure to do so is exceptional. The following categories are exhaustive:

| Situation | Classification | Required behavior |
| --- | --- | --- |
| The effective template exposes no ICN-supported reasoning behavior | Successful discovery | Return the normalized option list `none`, defaulting to `none`. |
| The canonical execution profile exceeds available resources | Successful discovery | Return hardware `DoesNotFit`. |
| An artifact is unreadable, truncated, malformed, or not GGUF | Invalid artifact | Do not return it as `Available`. Preserve a specific artifact diagnostic if invalid artifacts are exposed. Do not attach unknown properties. |
| The pinned runtime does not support the architecture, quantization, component combination, or required execution plan | Incompatible artifact | Do not return it as `Available`. Report a specific incompatibility at model availability level, not unknown reasoning or hardware. |
| A file disappears or changes during inspection | Concurrent mutation | Discard the attempt, reconcile identity, and retry from a stable snapshot. If stability cannot be established within the bounded retry policy, fail the ensure request. |
| Filesystem access, device enumeration, allocation-free native planning, template compilation, or another required dependency fails unexpectedly | Operational failure | Fail the ensure request with the underlying diagnostic. Do not persist a partial result or return a successful response containing unknown fields. |
| The inspector or estimator cannot derive a declared property from a valid, stable, runtime-supported model despite having the required inputs | ICN implementation defect | Treat the ensure operation as failed, preserve diagnostics, and fix the implementation. Do not normalize the defect into a product state. |
| The public contract asks for a semantic fact that artifacts and deterministic execution cannot establish | Contract defect | Narrow or redefine the contract. Do not add an `Unknown` state to conceal an unobservable property. |

Invalid and incompatible artifacts are expected to be rare and are not available models. Operational failures and ICN implementation defects are also not expected paths. They should be observable as errors and must not be cached as model facts.

If product requirements later demand partial availability when one artifact is bad, that behavior should be implemented through an explicit top-level invalid/incompatible artifact record. It must not weaken the property invariant for models labeled `Available`.

## Cache validity

A path existing in the cache is not evidence that its assessment remains valid. Reconciliation invalidates results when any relevant input changes, including:

- component membership, relationships, size, timestamps, or content identity;
- effective chat template, tokenizer, BOS/EOS tokens, or named-template selection;
- inventory schema or inspection algorithm version;
- pinned llama.cpp version, native build, backend, or estimator fingerprint;
- canonical execution profile or capacity-policy version;
- hardware and device topology relevant to planning.

Reasoning and hardware evidence may use separate cache keys because their inputs differ. Completed results are written atomically only after all required work for the model succeeds. Interrupted or failed inspection must leave the previous valid entry intact only when its key is still valid; otherwise the model remains stale and the list request fails rather than returning it as current.

## Contract consequences

The durable and wire contracts should represent domain outcomes, not internal progress. For available models:

- reasoning is a completed normalized option list with a contained default;
- hardware has only `Fits` and `DoesNotFit`;
- property inspection has a completed form, not `Pending`;
- inspection errors are operation errors or top-level artifact availability diagnostics.

Progress for downloads and other explicit operations remains in their operation/status contracts. If inspection progress needs observability, it belongs in logs, traces, or an operation-specific endpoint, not in a successful model inventory snapshot.

## Acceptance criteria

The implementation satisfies this design when:

- daemon startup can complete without an inventory-wide scan or native assessment;
- the first list after a cold start reconciles and fully assesses all available models;
- a warm list performs reconciliation and reuses only valid cached evidence;
- simultaneous lists share one ensure operation;
- adding, changing, regrouping, or removing artifacts invalidates exactly the affected entries;
- no successful response contains unresolved properties for an available model;
- missing template metadata follows runtime fallback semantics;
- reasoning inspection uses execution-equivalent token and template inputs;
- the reasoning `none` profile and hardware `DoesNotFit` can only result from successful inspection;
- corrupt or incompatible artifacts never masquerade as available models with unknown properties;
- unexpected native, environmental, or derivation failures fail reconciliation with actionable diagnostics;
- model loading independently validates the exact requested execution plan.
