---
applies_to:
  - packages/acn/src/handlers.ts
  - packages/acn/src/model-*.ts
  - packages/acn/src/provider-model-catalog.ts
  - packages/acn/src/service-operation-coordinator*.ts
  - packages/acn/src/session-*.ts
  - packages/acn/src/local-model-*.ts
  - packages/icn/src/downloads/**
  - packages/protocol/src/rpcs/local-inference.ts
  - packages/protocol/src/rpcs/session.ts
---

# Operation ownership

Transport lifetime and domain-operation lifetime are independent.

An RPC is classified by its semantics:

- a query is caller-owned and cancellation stops it;
- an observation is caller-owned and cancellation removes only that observer;
- a finite mutation is domain-owned after its linearization point and must then finish or roll
  back; and
- shared work is owned by its domain service after admission, so caller cancellation stops only
  that caller's acknowledgement or wait.

Expected duration does not determine the class.

## Shared operations

A domain service that publishes shared nonterminal state owns the operation which can terminalize
that state. Admission atomically validates current state, determines whether the request is already
satisfied, joins or rejects an active operation, acquires continuing ACN activity, publishes the
nonterminal state, and starts the owner in that service's scope.

Equivalent callers share one completion. Conflicting callers are rejected or serialized according
to domain policy. Interrupting a waiter cannot interrupt the owner or another waiter. Only an
explicit domain cancellation command may cancel admitted work while the owning service remains
alive.

Every owner handles its complete Effect `Exit`. Success, typed failure, defect, and interruption
all remove ownership and produce a terminal state before releasing the activity claim. A public
nonterminal state without a live owner is invalid.

Operation identity and completion may remain internal. Clients obtain current truth from
authoritative revisioned mirrors, including after reconnect; they do not reconstruct operations
from command responses, progress streams, or client-maintained timers. No global workflow registry
or client-visible operation ID is implied.

Owned fibers are children of the responsible service scope. Unscoped daemon fibers are not an
ownership model. Service teardown may interrupt them because teardown also destroys the authority
whose state they govern.

ACN implements these mechanics with a narrow service-operation coordinator owned by each
participating domain service. The coordinator owns one active semantic key, an interruptible
admission lock, equivalent-key joining, conflicting-key classification, a shared terminal outcome,
the ACN activity claim, and the service-scoped owner fiber. It does not own product state,
validation rules, conflict policy, operation persistence, or recovery.

The domain derives the requested key and validates current authoritative state while holding the
coordinator's admission lock. If idle work is needed, only the admission commit is masked:
continuing activity is acquired, ownership is recorded, the public nonterminal state is published,
and the owner is launched. Terminal product state is committed before the coordinator removes
ownership and releases the shared outcome. Thus callers cannot race a separate satisfaction check
against admission, and cancellation before the commit cannot admit unwanted work.

## Finite mutations

Validation before a mutation's linearization point remains interruptible. After that point, the
mutation either completes or rolls back within an uninterruptible domain transition, or hands
execution to a service-owned operation when the work can block for an unbounded period. Retry must
converge from every observable partial result. A short mutation does not gain a fabricated
operation state merely to manage cancellation.

## Domain applications

Model load, unload, and replacement are one serialized residency domain. Explicit commands and
chat preparation use the same coordinator. `LoadingLocalModel` and `UnloadingLocalModel` are valid
only while their matching coordinator owner exists. Chat preparation joins load completion and
holds residency admission until ICN accepts the generation lease; loading is never readiness.

Provider catalog refresh is a catalog-owned single flight. Equivalent callers join it, conflicting
targeted refreshes serialize, and every exit publishes a terminal catalog state.

Draft preload and claim phases are outcome-total. Cancellation removes a preloading record or
restores a claim. Claiming linearizes session creation: after the claim, initial-event publication
and promotion or rollback form one outcome-total mutation independent of client interruption.

ICN owns accepted download attempts durably. ACN observes attempts independently of the initiating
request, polling quickly while active and periodically while idle so missed request-side wake-ups
cannot hide shared work.

## Conformance

- Disconnecting the initiating client during load, unload, or refresh cannot strand nonterminal
  state.
- Equivalent commands from multiple clients cause one effective operation and expose the same
  snapshots.
- A client connecting after completion receives the terminal snapshot.
- Interrupting draft preload or claim leaves no orphaned phase.
- An accepted native download becomes observable without further initiating-client action.
- Tests assert only current intended states and never preserve a legacy stranded state.
