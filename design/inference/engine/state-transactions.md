---
applies_to:
  - inference/engine/state/**
  - inference/engine/executor/**
  - inference/engine/generation/**
  - inference/engine/scheduler/**
---

# Numerical state transactions

Tentative numerical state belongs to an owned advance. It carries the accepted source state,
successor reservations, device binding views, and proposed extent. The advance can move into a
validated launch and remain in flight without borrowing a sequence record. Dropping unfinished
work releases every tentative claim. An explicit abort recovers the unchanged accepted source
state when the request can continue.
Recurrent state lives in bank slabs; a bank contains one conversation's recurrent-layer state
and tape. Banks are claimed and returned through shared ownership exactly as separate allocations
would be.
An advance names its accepted bank and its successor bank, and the batch carries both per slot, so
kernels read one row and write another in place. No kernel writes an accepted bank or the zero
seed; forks and checkpoints share accepted banks by claim, never by copy.
Attention history rows share history slabs. Each accepted history is an ordered list of spans;
one span is contiguous within one slab and spans need not ascend in address order. An advance
first grows its history into free rows after its last row within that slab. Otherwise it continues
in another free span or a new slab. A fresh history begins within the largest free span, leaving
room for the history ending before it; a span beginning at row zero fills from its start. A free
span never crosses a slab boundary. Per-row references preserve shared prefixes and checkpoints.
The store compacts before a launch that would exceed its span bound, computed for the loaded model
as `ceil(context limit / rows per slab) + 16`.

Each history slab contains every component at a fixed aligned offset. Dense components hold
activation values; affine components hold codes and scale/zero coefficients. A row has the same
slab index and offset in every component, so placement, compaction and conversion move all parts
of that row together. A slab targets 64 MiB: its row count is rounded down to a multiple of the
256-row tile, with at least one tile. Bank slabs hold as many complete banks as fit in that target,
with at least one bank.

Elastic state growth is admitted before an advance. The store claims one required slab through
the heap before adding it, without replacing or copying existing slabs. A refused claim or slab
allocation leaves accepted numerical state, published backing, and committed charge unchanged;
growth that needs both history and bank slabs is one fallible operation. The refusal returns an
explicit memory deficit to the request owner for reclamation and retry. Empty slabs release their
measured charge without a new claim.
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
submit, complete, finish, reconcile lifecycle. Compaction runs only while the store has no
transaction, moves rows and banks into free space of slabs already held after submitted writes
complete, and publishes the rewritten histories and bank placement only after every copy succeeds.
It merges spans before the span bound is exceeded and empties the least occupied slabs under
pressure. Accepted state keeps its logical identity throughout. Cancellation, submission failure,
device failure and teardown release reservations through ownership.
Preemption releases physical request state while preserving accepted logical tokens. Restoration
replays only to the numerical position that existed before eviction. An accepted successor that
has not yet been consumed numerically remains the input for the next ordinary decode; replay must
not consume it early or advance beyond that numerical boundary.

Codec conversion reserves destination history before submission. One transaction owns both
stores' sequence claims, their source and destination slabs, both recurrent banks, and a
per-layer key/value mapping between codec components. The mapping names the source and
destination codecs and row addresses; it is validated against a fresh destination, compatible
layer widths and recurrent layout, and the selected stores. Abort returns both unchanged states.
Commit publishes the destination position and history only after the state program finishes.

## Acceptance criteria

- No in-flight state transaction borrows sequence storage.
- Interleaved advances of concurrent sequences add no history span while the following rows in
  the same slab are free; a span never crosses a slab boundary.
- Every history fits the loaded model's span bound, including at full context and after a freed
  slab index is reused.
- Compaction publishes only after all copies complete, preserves greedy continuation, and needs
  no new memory claim on any backend.
- Failed joint history and bank slab growth restores both published backings and their charge.
- A new sequence observes zero recurrent state even after prior sequences have returned dirty banks.
- A successor bank is never the zero seed, an accepted bank a live state, checkpoint or fork can
  read, or another in-flight successor.
- Submit failure and cancellation leave accepted state unchanged and release tentative claims.
- Full, partial, and zero acceptance reconcile each transaction exactly once.
- A recurrent interior prefix is not visible until its repair work completes.
- An accepted/resident checkpoint is created only after reconciliation.
