---
applies_to:
  - packages/acn/src/service-operation-coordinator*.ts
  - packages/acn/src/boundary/**
  - packages/acn/src/session-*.ts
  - packages/acn/src/model-*.ts
  - packages/icn/src/**
  - inference/crates/icn-server/**
  - packages/sdk/src/jit-rpc/**
  - packages/acn-protocol/src/boundary/**
---

# Operation ownership

Transport lifetime and domain-operation lifetime are independent. Duration does not determine
ownership.

| Operation | Owner | Effect of caller cancellation |
| --- | --- | --- |
| Query | Caller | Stops the query |
| Observation | Caller | Removes only that observer |
| Finite mutation before commit | Caller | Prevents the mutation |
| Finite mutation after commit | Domain transition | Must complete or roll back |
| Shared admitted work | Domain service or resource generation | Stops only that caller's wait |

## Admission and concurrency

A service that publishes shared nonterminal state owns the work capable of terminalizing it.
Admission is one atomic decision over current authoritative state:

```text
validate -> already satisfied / join / reject / serialize
         -> acquire continuing lifetime
         -> record ownership and publish nonterminal state
         -> start owner in the service scope
         -> acknowledge admission
```

Cancellation before the commit admits nothing. Afterward, equivalent callers share one outcome and
interrupting one waiter cannot interrupt the owner or another waiter. Conflicting requests are
rejected, serialized, or superseded only by explicit domain policy. Only a domain cancellation
command may cancel admitted shared work while its authority remains alive.

The requested key and satisfaction check are evaluated inside the admission boundary; callers
cannot race a separate observation against admission. Masking covers only the commit that
establishes lifetime, ownership, public state, and the owner fiber. Potentially unbounded work runs
interruptibly in the owning scope.

## Terminalization

Every owner handles its complete Effect `Exit`: success, typed failure, defect, interruption, and
owner shutdown. Terminal product state is committed before ownership and continuing lifetime are
released. Public nonterminal state without a live owner is invalid.

Continuing lifetime is local to the authority whose state remains nonterminal. A session's
aggregate `Working` status is already authoritative agent state: its loaded `SessionRuntime`
observes that status directly instead of manufacturing a second continuing-work claim. This does
not retain the ACN process and is not duplicated per message, worker, detached process, or observer.

Owned work is a child of the responsible service scope, never an unscoped daemon fiber. Service
teardown may interrupt it because teardown destroys the authority whose state it governs, but the
same teardown must terminalize or remove that state. Process shutdown is additionally single-flight:
once `Stopping` commits, interruption of the requester cannot abandon TERM/KILL/reap work.

Operation identity may remain internal. Clients recover current truth from authoritative queries;
they do not reconstruct ownership or completion from command responses, progress streams, timers,
or a universal workflow registry.

## Finite mutations

Validation before a mutation's linearization point remains interruptible. After that point, the
mutation completes or rolls back inside an outcome-total transition, or atomically hands continuing
work to a service owner. Retry must converge from every observable partial result. Short mutations
do not gain fabricated operation state merely to simplify cancellation.

Ambiguous transport failure does not authorize generic mutation replay. Replay requires durable
domain idempotency; otherwise the client reconciles the unknown outcome from authoritative state.

## Conformance

- Disconnecting an initiating client cannot strand public nonterminal state.
- Equivalent admitted requests perform one effective operation and expose one outcome.
- A later client observes current terminal state without the original request.
- Caller cancellation before admission cannot start unwanted work.
- Every admitted operation retains its owner and lifetime until terminal publication.
- Teardown cannot leave owned work detached from the authority it governed.
