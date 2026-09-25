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
Recurrent state lives in one arena per component and layer; a bank is an index into those
arenas, claimed and returned through shared ownership exactly as a separate allocation would be.
An advance names its accepted bank and its successor bank, and the batch carries both per slot, so
kernels read one row and write another in place. No kernel writes an accepted bank or the zero
seed; forks and checkpoints share accepted banks by claim, never by copy.
Attention history rows are one shared arena, and each accepted history is a list of address
ranges (segments) in logical order; the attention entries bound the segments a row may read.
Placement keeps that count independent of how requests interleave: an advance first grows its
sequence in place, into the free rows that begin at its history's end, so rows released by a
rejected tail or an abort are reused in place. Rows that cannot grow in place (a fresh sequence, a
fork whose sibling took the rows, a neighbouring history) start at the middle of the largest free
run, leaving the rows before them as growth room for the history that ends there; a run starting
at row 0 has no such history and fills from its start. Only a reservation larger than every free
run splits across runs, largest first. Segment addresses need not ascend in logical order.
Every history plane is indexed by that same row, with one codec group per (row, kv head) vector:
a plane is `[rows, kv heads, elements]`. Dense history has one activation plane per vector kind;
affine history has a code plane and one coefficient plane of (scale, zero) pairs, so placement,
compaction and conversion treat every codec's planes alike.
Elastic backing growth is admitted before an advance. Its minimum is the rows
and successor banks the launch needs, including relayout when fragmentation
prevents the demanded history from growing contiguously; its preferred grant includes geometric
headroom and optional relayout. The claim covers the peak Seismic charge,
including the old and new backing held together during reallocation. If the
preferred grant cannot fit, the store tries the minimum. A refused minimum
leaves accepted numerical state intact and returns an explicit memory deficit
to the request owner for reclamation and retry.
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
Preemption releases physical request state while preserving accepted logical tokens. Restoration
replays only to the numerical position that existed before eviction. An accepted successor that
has not yet been consumed numerically remains the input for the next ordinary decode; replay must
not consume it early or advance beyond that numerical boundary.

Codec conversion reserves destination history before submission. One transaction owns both
stores' sequence claims, all source and destination plane buffers, both recurrent banks, and a
per-layer key/value mapping between physical codec planes. The mapping names the source and
destination codecs and row addresses; it is validated against a fresh destination, compatible
layer widths and recurrent layout, and the selected stores. Abort returns both unchanged states.
Commit publishes the destination position and history only after the state program finishes.

## Acceptance criteria

- No in-flight state transaction borrows sequence storage.
- Interleaved advances of concurrent sequences add no history segment while a sequence's following
  rows are free; lock-step serving with speculative tails, completions and admissions keeps every
  history within two segments even when the arena holds exactly one context per request.
- A new sequence observes zero recurrent state even after prior sequences have returned dirty banks.
- A successor bank is never the zero seed, an accepted bank a live state, checkpoint or fork can
  read, or another in-flight successor.
- Submit failure and cancellation leave accepted state unchanged and release tentative claims.
- Full, partial, and zero acceptance reconcile each transaction exactly once.
- A recurrent interior prefix is not visible until its repair work completes.
- An accepted/resident checkpoint is created only after reconciliation.
