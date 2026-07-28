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

Domains that need shared ACN work may implement these mechanics with the narrow
service-operation coordinator. It is not a universal operation registry.

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

Model load, stop, replacement, worker supervision, and inference leasing are owned by ICN's
`ModelInstanceController`. ACN's `ModelSlotController` submits finite idempotent commands and
projects `ModelInstancesSnapshot`; it does not own a second physical operation or admission gate.
Loading and Stopping are valid because their exact native instance has a live ICN owner.

The slot controller does own the finite command-admission boundary: equivalent concurrent load
requests serialize, re-read canonical slot state, and share an already Loading or Ready instance.
This prevents two callers from deriving distinct native IDs from one loadable slot observation
without making ACN an owner of the physical load operation.

One branded model-instance identity is created before load admission and remains unchanged through
loading, readiness, and stopping. Stop is keyed only by that identity; slot and provider-model
identity are intentionally absent because they would be redundant and could disagree. Absence of
an instance is represented with `Option`, never `undefined`.

Stop enters the exact instance's native transition. Unknown and terminal IDs are idempotent.
Loading transitions through Stopping, cancels its owned operation, reaps its owned worker, and
becomes Stopped. Ready closes exact lease admission, drains accepted leases, reaps its worker, and
becomes Stopped.

Slot reconciliation cannot author physical lifecycle. It retains durable selection and directly
projects the exact bound native instance. Catalog and provider changes affect availability; they
cannot overwrite instance lifecycle.

Local slot assignment validates the exact installed serving configuration and commits that durable
selection. Slot actions remain presentation and are never used as command authorization.
Assignment and load never invoke the preview operation; ICN load admission is the only
authoritative load-time hardware decision.

Provider catalog refresh is a catalog-owned single flight. Equivalent callers join it, conflicting
targeted refreshes serialize, and every exit publishes a terminal catalog state.

Draft preload and claim phases are outcome-total. Cancellation removes a preloading record or
restores a claim. Claiming linearizes session creation: after the claim, initial-event publication
and promotion or rollback form one outcome-total mutation independent of client interruption.

ICN owns accepted download attempts durably. ACN observes attempts independently of the initiating
request, polling quickly while active and periodically while idle so missed request-side wake-ups
cannot hide shared work.

A completed attempt remains successful while installed inventory converges. ACN refreshes and waits
for that authority instead of interpreting the absence of an active attempt as failure.

Onboarding is a client-owned composition of ordinary model commands. An explicit choice waits for
the target-level download command to observe authoritative installation, assigns the offering to the
primary slot, invokes the ordinary slot load, and marks onboarding complete only after that load
observes the exact selected instance as Ready. ACN owns only the generic commands and the durable
onboarding boolean; it has no onboarding-specific model operation, activation service, or startup
reconciler. The client retains only its submitted choice while mounted. Cancellation uses the
ordinary target-download or slot-clear command. Interruption or restart never reconstructs command
intent from onboarding, download, assignment, or instance snapshots.

## Conformance

- Disconnecting the initiating client during load, stop, or refresh cannot strand nonterminal
  state.
- Equivalent commands from multiple clients cause one effective operation and expose the same
  snapshots.
- A client connecting after completion receives the terminal snapshot.
- Interrupting draft preload or claim leaves no orphaned phase.
- An accepted native download becomes observable without further initiating-client action.
- Starting ACN or reopening onboarding never turns a saved slot assignment into a model load.
- Tests assert only current intended states and never preserve a legacy stranded state.
