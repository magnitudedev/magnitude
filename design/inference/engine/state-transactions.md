---
applies_to:
  - inference-v4/engine/model-state/**
  - inference-v4/engine/model-executor/**
  - inference-v4/engine/generation/**
  - inference-v4/engine/service/**
---

# Numerical state transactions

Tentative numerical state belongs to an owned advance. It carries the accepted source state,
successor reservations, device binding views, and proposed extent. The advance can move into a
validated launch and remain in flight without borrowing a sequence record. Dropping unfinished
work releases every tentative claim. An explicit abort recovers the unchanged accepted source
state when the request can continue.
Every newly created sequence begins from one immutable, pristine zero recurrent bank. It may share
that seed with other new sequences; the first and every later advance reserves a distinct writable
successor. Returned successor banks never become the initial state of another sequence. The zero
seed is included in the planned persistent state charge.

Submission does not make successor state visible. Physical completion produces an outcome that
still owns the advance. Generation prepares a logical acceptance decision without mutating its
live record. Reconciliation consumes the physical outcome and commits exactly the accepted prefix
or aborts it. Only after successful reconciliation does generation apply its prepared transition.
Preparation forks the request-local generation method, evaluates grammar and method effects,
stabilizes transient method features through an owned retainer, checks all counters and output
indices, and stores the resulting logical state in one owned
`PreparedGenerationTransition`. The executor receives only its `ReconcileDecision`. Applying
the transition after physical reconciliation has no recoverable errors. A cancelled or failed
physical reconciliation drops the staged method and grammar without changing the live request.

An interior accepted prefix with recurrent state requires numerical repair before the successor
can be published or checkpointed. State compaction, copying, and codec conversion follow the same
submit, complete, finish, reconcile lifecycle. Cancellation, submission failure, device failure,
and teardown release reservations through ownership.

Codec conversion reserves destination history before submission. One transaction owns both
stores' sequence claims, all source and destination plane buffers, both recurrent banks, and a
per-layer key/value mapping between physical codec planes. The mapping names the source and
destination codecs and row addresses; it is validated against a fresh destination, compatible
layer widths and recurrent layout, and the selected stores. Abort returns both unchanged states.
Commit publishes the destination position and history only after the state program finishes.

## Acceptance criteria

- No in-flight state transaction borrows sequence storage.
- A new sequence observes zero recurrent state even after prior sequences have returned dirty banks.
- Submit failure and cancellation leave accepted state unchanged and release tentative claims.
- Full, partial, and zero acceptance reconcile each transaction exactly once.
- A recurrent interior prefix is not visible until its repair work completes.
- An accepted/resident checkpoint is created only after reconciliation.
