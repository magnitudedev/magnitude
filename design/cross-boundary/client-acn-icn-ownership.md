---
applies_to:
  - inference/**
  - packages/icn/**
  - packages/icn-protocol/**
  - packages/acn/**
  - packages/acn-protocol/**
  - packages/agent/src/ambient/config-ambient.ts
  - packages/sdk/**
  - packages/client-common/**
  - cli/**
  - desktop/**
---

# Client, ACN, and ICN ownership

This document defines the semantic boundaries between inference, the Magnitude application
backend, shared client infrastructure, and individual clients. It governs both operations and
data contracts crossing those boundaries.

## Layer model

```text
operating systems, drivers, and native runtimes
                         |
                         v
              ICN inference truth
                         |
                         v
              ACN application truth
                         |
                         v
          SDK and client-common transport
                         |
                         v
                 client presentation
```

Each boundary is an abstraction boundary. A lower layer exposes the smallest stable semantic
contract its caller actually needs; it does not expose implementation-specific inputs merely
because they were used to produce that contract.

Each semantic decision has one owner. A higher layer may project an owned result into its own
contract, but it must not independently recreate the decision or redefine the lower-layer domain
shape.

For model presentation, ICN publishes reviewed variant facts with each recommendable configuration.
ACN owns the unified local presentation projection and carries the variant label independently from
the base display name, exact quantization, and precision. The ACN protocol exports the canonical
`Name (Variant)` composition over those structured fields so agent attribution and clients cannot
diverge; client-common exposes it to clients. Individual clients own width-aware layout and
truncation. No boundary parses a display string or infers a label from fidelity rank.

## ICN

ICN owns inference truth and the resources that make inference possible:

- hardware discovery and normalized hardware observations;
- native model inspection, configuration, planning, and assessment;
- model compatibility and inference feasibility;
- inference safety thresholds and runtime admission;
- model-instance allocation and lifecycle;
- serving-time supervision, pressure response, and release; and
- structured factual evidence from attempted inference operations.

ICN contracts describe inference-domain facts and outcomes. They may be used by any application
without knowing Magnitude's screens, recommendation vocabulary, or presentation policy.

ICN does not own:

- Magnitude recommendations or preferred resource usage;
- warning categories such as `Tight fit`;
- onboarding or menu concepts;
- application-specific status precedence;
- copy, formatting, or interaction behavior; or
- convenience operations whose purpose is to evaluate application guidance.

### Platform normalization

Operating-system and runtime differences terminate inside ICN unless the difference is itself a
meaningful inference-domain fact required by callers.

For example, Windows commit availability is an input to system-allocation safety. It is not a
separate concept that Magnitude clients need to understand. ICN combines every applicable native
limit into a common system-allocation contract:

```text
SystemMemoryObservation
  physicalCapacityBytes
  physicalAvailableBytes
  allocationCapacityBytes
  allocationHeadroomBytes
```

`physical*` describes physical system memory for hardware presentation. `allocation*` describes
the effective capacity and current headroom available to an inference system-memory allocation
after all applicable platform limits are considered.

On systems without an additional allocation limit, allocation capacity and headroom equal the
physical values. On Windows, ICN accounts for both physical memory and commit internally and
publishes the binding normalized values. Windows-specific commit fields and variants do not cross
the ICN contract.

This rule applies generally: platform adapters may be different; the public semantic contract is
common.

### Safety and authority

ICN owns thresholds that change whether inference fits, may start, or may continue. Assessment
reserves and runtime abort reserves therefore belong to ICN. Callers select the model and serving
profile; they do not supply the meaning of `Fits` or the safety boundary for loading.

An observation or preview is advisory. The ICN operation that creates or changes an inference
resource must revalidate its authoritative preconditions at the mutation boundary.

## ACN

ACN owns Magnitude application semantics:

- product policy and recommendations;
- application interpretations of ICN facts;
- correlations across inference, configuration, storage, and session state;
- durable user intent and application operation lifecycles;
- disposable product projections consumed by clients; and
- the complete client-facing application contract.

When a client needs a conclusion that combines multiple backend facts, ACN performs the join once
and publishes the result. Clients do not independently join raw hardware, assessment, and instance
mirrors to reconstruct application state.

ACN may define advisory policy, such as how much system headroom Magnitude recommends leaving for
other applications. Such policy may produce warnings, ranking inputs, or explanations, but it
cannot authorize an ICN load or redefine an ICN assessment.

ACN does not:

- duplicate native planning or inference safety calculations;
- treat a copied ICN observation as authority over an ICN resource;
- ask ICN to evaluate a Magnitude-specific warning or screen state; or
- expose raw ICN wire types directly to clients.

### Application projections

Derived product state belongs to ACN when it combines backend facts or applies Magnitude policy.
It is represented as an ACN protocol schema and, when reactive, as a mirrored state with one
semantic lifecycle.

Stable and volatile evidence may share one atomic product row when clients need their relationship.
Their semantic substructures still retain independent invalidation rules: changes in live hardware
pressure update current headroom without changing stable model assessment or recommendation
identity. A separate mirror is justified only for a separately consumable domain authority, not
merely because one nested value changes more frequently.

Derived application state is advisory unless its owning operation is also the authoritative
mutation owner. Advisory state never becomes an authorization token for a later mutation.

## Protocol and SDK boundaries

The ICN contract contains normalized inference-domain schemas and operations. Platform-specific
mechanics remain beneath it.

The ACN protocol contains complete Magnitude application schemas and operations. It may project
ICN facts into application shapes, but private ICN types and application-irrelevant native details
do not leak through it.

The SDK provides typed transport, daemon lifecycle, and access to the ACN protocol. It does not
own product policy or create a parallel domain model.

At each real serialized boundary, a concept has one canonical shape. A projection into a new
domain is justified only when the receiving layer gives the data different semantics. Copying a
shape into another package for convenience is not a new domain and is prohibited.

## Client-common

Client-common owns shared client infrastructure:

- typed access to SDK operations and mirrored states;
- reactive query, mutation, invalidation, and subscription behavior;
- reusable hooks and identity-safe selectors;
- shared interaction infrastructure; and
- presentation-neutral utilities that are genuinely common across clients.

Client-common imports application contracts through the SDK. It does not redefine ACN or ICN
domain unions, calculate backend policy, or become a second application backend in the client
process.

For an Effect Query-adopted subsystem, client-common owns the canonical query definitions,
mutation definitions, semantic mutation scopes, and the invalidation bridge from ACN mirror events.
The bridge treats events only as notification and rereads the ACN snapshot. Effect Query mutation
states describe exact client command invocations; they do not duplicate ACN installation or
ICN download lifecycle state.

Client-common must not:

- define parallel memory-assessment, fit, guidance, or loadability shapes;
- calculate assessment, admission, reserve, recommendation, or warning policy;
- join independent backend mirrors to manufacture a new product fact;
- parse diagnostic prose into structured state; or
- retain copied backend facts as an independent authority.

A pure selector may locate an already-owned entry by its complete identity. It may not change the
entry's meaning or silently join it to superseded evidence.

## Individual clients

Individual clients own presentation and interaction:

- wording and explanatory copy;
- byte, duration, and number formatting;
- colors, layout, icons, and responsive behavior;
- local ephemeral interaction state; and
- submitting typed application operations.

Clients render ACN states exhaustively. They do not infer domain results from missing fields,
recalculate backend policy, or parse server messages for facts that should have been structured.

Presentation precedence may determine which label occupies limited space, but it must not erase
independent domain evidence. A recommendation label and a memory warning, for example, can remain
separate even when one is visually more prominent.

## Operation placement

An operation belongs to the layer that owns the resource or irreversible state it changes.

- An ICN mutation changes inference resources and revalidates inference safety.
- An ACN mutation changes Magnitude state or coordinates a Magnitude lifecycle.
- A client interaction submits one of those operations; it does not become the operation owner.

Queries observe the owner's state. A new lower-layer RPC is not justified merely because a higher
layer wants a convenient application conclusion. If ACN can derive a Magnitude result from
existing ICN facts, that derivation belongs in ACN.

Batching follows the same rule. A batch ICN operation is appropriate only when batching is part of
an inference-domain operation or provides consistency ICN owns. It is not appropriate as a vehicle
for moving application evaluation into ICN.

Callers may rely on an observational result only for the observation it describes. A preview does
not authorize a later mutation, and an actual mutation always checks the owner's current state.

## Ownership rubric

| Question | Owner |
|---|---|
| Does it require native, platform, hardware, or inference truth? | ICN |
| Does it change inference feasibility, admission, allocation, or safety? | ICN |
| Does it combine backend facts or apply Magnitude product policy? | ACN |
| Is it Magnitude durable intent or an application operation lifecycle? | ACN |
| Is it typed transport, reactivity, or an identity-preserving shared selector? | SDK or client-common |
| Is it wording, formatting, layout, or local interaction state? | Individual client |

When ownership is unclear, separate these dimensions:

- fact versus interpretation;
- authoritative decision versus advisory projection;
- stable evidence versus volatile observation;
- operation versus observation; and
- domain semantics versus presentation.

Shared input data does not imply shared ownership of the decisions made from it.

## Memory example

The local-model memory flow illustrates the boundary:

```text
native OS observations
        |
        v
ICN normalized allocation capacity/headroom
ICN assessment and abort reserves
ICN Fits / DoesNotFit and actual load rejection
        |
        v
ACN recommended-headroom policy
ACN high-use and current-shortfall guidance
ACN atomic LocalModel projection
        |
        v
client-common mirrored-state access
        |
        v
CLI wording and layout
```

ICN retains Windows commit accounting internally. ACN receives normalized allocation headroom and
does not branch on the operating system. ACN owns `High memory use` and current advisory shortfall
and publishes them on the assessed local model rather than through a parallel guidance mirror.
The actual load remains authoritative in ICN and may still reject after an earlier advisory
observation. A structured low-memory failure reports the common allocation boundary and shortfall,
not the platform mechanism that happened to bind it.

## Prohibited patterns

- An ICN operation named after onboarding, a menu, or a Magnitude warning.
- An ACN request asking ICN whether Magnitude should display a warning.
- A client-common union parallel to an ACN protocol union.
- A client joining assessment, hardware, and instance state to reconstruct product guidance.
- A client parsing error prose to recover required bytes, reserve, or shortfall.
- The same policy formula implemented in ICN, ACN, and a client under one meaning.
- Platform-specific fields crossing ICN when callers need only a normalized semantic result.
- An advisory projection being treated as permission for an authoritative mutation.
- A copied observation being persisted or versioned as a second source of truth without a distinct
  semantic lifecycle.

## Conformance

- Every semantic decision has one named owner.
- Platform-specific inference mechanics terminate at the ICN adapter unless they are meaningful to
  all callers as domain facts.
- ICN contracts contain no Magnitude recommendation, warning, or presentation policy.
- ACN application policy cannot alter ICN truth except through an explicit ICN mutation.
- ACN publishes client-ready domain state for conclusions that require backend joins.
- Client-common contains no duplicated backend domain schemas or policy calculations.
- Clients can render complete states without parsing diagnostic prose or joining raw backend facts.
- Advisory observations are identified as advisory, and authoritative operations revalidate.
- Derived state changes only when its semantic inputs change and retains the identities needed to
  reject stale joins.
- Structured failures preserve the producing operation's factual evidence without leaking private
  platform mechanisms.
