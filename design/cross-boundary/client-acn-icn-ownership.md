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
clients -> one Effect Query runtime -> ACN application RPC
                                  \-> ACN /inference proxy -> ICN inference API
```

Application RPC uses the current owner-record endpoint. The inference proxy is exposed on ACN's
stable public listener at `127.0.0.1:10100`; these are two listeners of one admitted ACN process,
not two services or authorities.

Each boundary is an abstraction boundary. A lower layer exposes the smallest stable semantic
contract its caller actually needs; it does not expose implementation-specific inputs merely
because they were used to produce that contract.

Each semantic decision has one owner. A higher layer may project an owned result into its own
contract, but it must not independently recreate the decision or redefine the lower-layer domain
shape.

ICN publishes model, package, download, instance, and hardware resources. ACN publishes only
Magnitude application resources. The shared client may compose these independently owned resources
for presentation, but it does not copy that composition into a second writable authority.

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
- Magnitude-specific provider and recommendation projections; and
- the complete client-facing application contract.

ACN does not proxy individual inference DTOs through RPC. It owns one transparent streaming prefix
proxy and injects the private ICN authorization. The public OpenAPI document is ICN's document with
the proxy base path.

ACN's Magnitude-specific projections observe their private ICN through one shared multiplexed
resource-event connection. Each projection subscribes before reading its authoritative snapshot;
on invalidation or reconnect it rereads that snapshot. Hardware consumers additionally poll because
memory used by unrelated operating-system processes has no ICN mutation event. Individual ACN
projections must not open separate ICN event streams or invent other per-resource polling loops.
The event endpoint emits one current invalidation per selected topic when a connection opens, so
first-party query drains also converge if their initial read races stream establishment.

ACN may define advisory policy, such as how much system headroom Magnitude recommends leaving for
other applications. Such policy may produce warnings, ranking inputs, or explanations, but it
cannot authorize an ICN load or redefine an ICN assessment.

ACN does not:

- duplicate native planning or inference safety calculations;
- treat a copied ICN observation as authority over an ICN resource;
- ask ICN to evaluate a Magnitude-specific warning or screen state; or
- reinterpret ICN resource types as ACN application resources.

Slots retain durable selection only. Recommendations remain ACN application policy keyed by the
same canonical model ID. ACN never mirrors Instance residency into Slots. Its read-only Local
Models projection is permitted because it gives native facts Magnitude-specific assessment,
provider, recommendation, and warning semantics; it owns no inference transition and is not a
copy of the native Models contract.

## Protocol and SDK boundaries

The ICN contract contains normalized inference-domain schemas and operations. Platform-specific
mechanics remain beneath it.

The ACN protocol contains Magnitude application schemas and RPC operations. ICN Rust routes and
schemas generate the canonical inference OpenAPI contract. The SDK derives one low-level
TypeScript inference client from that generated contract and authors an Effect Query group over it.
Adding an inference operation requires no ACN RPC handler or proxy generation.

At each real serialized boundary, a concept has one canonical shape. A projection into a new
domain is justified only when the receiving layer gives the data different semantics. Copying a
shape into another package for convenience is not a new domain and is prohibited.

Electron `contextBridge` is one such serialized boundary. Values crossing it use the encoded form
of the existing canonical schema and are decoded immediately in the receiving process; the bridge
does not introduce a second domain contract.

## Client-common

Client-common owns shared client infrastructure:

- one connection-scoped Effect Query client combining ACN RPC and inference HTTP operations;
- reactive query, mutation, invalidation, and subscription behavior;
- reusable hooks and identity-safe selectors;
- shared interaction infrastructure; and
- reusable presentation primitives that are genuinely common across clients and contain no
  backend policy, such as memory, storage, and transfer byte-unit formatting.

Client-common imports both contracts through the SDK. It consumes ACN Local Models for product
semantics, ACN Slots for durable selection, and native Models and Instances for authoritative
installation and residency operations. Its final rendered view is a pure reactive join with no
writable copy and no inference admission decision.

ACN change pokes invalidate ACN Queries. ICN's multiplexed resource event stream invalidates native
Queries. Both are notification only; Query state is refreshed from its owning authority. Effect
Query mutation states describe exact invocations and never duplicate Download or Instance state.

Client-common must not:

- define parallel memory-assessment, fit, guidance, or loadability shapes;
- calculate assessment, admission, reserve, recommendation, or warning policy;
- join independent facts to recreate backend policy or authority;
- parse diagnostic prose into structured state; or
- retain copied backend facts as an independent authority.

A pure selector may locate an already-owned entry by its complete identity. It may not change the
entry's meaning or silently join it to superseded evidence.

## Individual clients

Individual clients own presentation and interaction:

- wording and explanatory copy;
- choosing and composing shared byte, duration, and number presentation primitives;
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
ACN Local Models product projection
        |
        v
client-common joins Slots and native Instance state
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
- A client joining assessment, hardware, and instance state to recreate admission policy.
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
- Each authority publishes complete owned resources; client-common may join identities for display.
- Client-common contains no duplicated backend domain schemas or policy calculations.
- Clients can render complete states without parsing diagnostic prose or recreating backend policy.
- Advisory observations are identified as advisory, and authoritative operations revalidate.
- Derived state changes only when its semantic inputs change and retains the identities needed to
  reject stale joins.
- Structured failures preserve the producing operation's factual evidence without leaking private
  platform mechanisms.
