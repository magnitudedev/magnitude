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
would be. A bank's components derive from its layers: convolution windows, gated delta states,
Mamba-2 state-space states (always F32) and their tapes, as an ordered list of sublayer banks per
block. A block's n-gram input keeps two windows after its mixer's components: an F32 window of its
dilated convolution's reach, and a token window of the sequence's tokens before the state (as many
as its row rule reads, stored token + 1 in U32 so the zero seed holds none). The token window is
how checkpoints, forks, prefix reuse and rollback carry the n-gram rule's preceding tokens; nothing
else holds them on the device. The executor keeps each resident sequence's tokens on the host to
the same rule: a target forward at a position replaces every token from that position on (rejected
verification rows, a discarded lookahead and a rollback are all fed again), and a resume state
carries the tokens before its position into the sequence that opens from it. A bank publishes its state after an
advance's committed rows and records every later row on its tape: the input rows of each
convolution window, the gated delta rule's innovations, keys and decays, and the Mamba-2 additive
rule's inputs, `B` rows and steps. A reader of version (bank, tape rows) replays those rows onto the
published state. Bank bytes, and with them the banks per slab, follow from the components.
An advance names its accepted bank and its successor bank, and the batch carries both per slot, so
kernels read one row and write another in place. A layer's components may be published by more
than one of its entries in the layer's ordered submission (a convolving input projection publishes
the successor window, its state entry the state and tape); each reads only the accepted bank and
writes only the successor. No kernel writes an accepted bank or the zero seed; forks and
checkpoints share accepted banks by claim, never by copy.
Attention history is organized in history domains. A history domain is a set of attention layers
whose history shares one row numbering; a row is one token's history in that domain's layers. A
store holds one history slab tensor per stored domain, each with its own components, rows per slab,
span bound, free space and per-row references. A domain is one of:

- **Token**: one row per token for the whole context.
- **Window(n)**: one row per token, but a history references only its last `n` accepted rows
  plus its tentative rows.
- **Shared**: no storage; its layers bind the regions of a source layer in a stored domain and
  see that domain's spans.
- **Block(rate)**: one row per `rate` tokens: block `b`'s row is written by the advance whose rows
  reach position `(b + 1)·rate`, and a history at position `p` references blocks `[0, p/rate)`.

A model has one Window domain per distinct window size. Qwen's stores have one Token domain.

A selected attention read's indexer (`LayerRef::Indexer`) keeps its raw keys, each with its row's
rotary coordinates, in a Window(rate − 1) domain and its finished block keys in a Block(rate)
domain; these domains hold indexer keys alone and follow every domain attention reads. Each is one
history read of a launch row: a row reads, as stored spans, the raw keys of the `rate − 1` positions
before it and every block complete before it (its advance's earlier rows and blocks at their
destinations), and writes its own raw key and the block it completes. An advance reserves one Block
row per block it completes; its commit of an accepted prefix keeps the blocks that prefix
completes; a checkpoint at `p` references raw keys `[p − rate + 1, p)` and blocks `[0, p/rate)`. So
prefix reuse, verification and rollback across a block boundary hold the blocks a straight run
does.

Each accepted history holds, per stored domain, an ordered list of spans; one span is contiguous
within one slab of its domain and spans need not ascend in address order. An advance of `n` rows
reserves the rows it adds in every stored domain of its store (`n`, or one per completed block in a
Block domain), as one transaction: a refusal in any domain
releases the rows reserved in the others. In each domain it first grows its history into free rows
after its last row within that slab. Otherwise it continues in another free span or a new slab. A
fresh history begins within the largest free span, leaving room for the history ending before it; a
span beginning at row zero fills from its start. A free span never crosses a slab boundary. Per-row
references preserve shared prefixes and checkpoints.

A Window(n) history writes every row of an advance, since the advance's later rows attend to its
earlier ones even when the advance is longer than the window. After each accepted advance at
position `p` it releases its references on rows at positions `< p − n`; its tentative rows are held
by the advance until reconciliation. Released rows are ordinary free space, and a slab left without
a referenced row is freed by the release rule below; no other release mechanism exists. A
checkpoint or retained prefix at position `p` references rows `[p − n, p)`, so forks and resumed
requests share them without copying, and an advance of a fork trims only the fork's references.
A successor formed before its predecessor reconciles reads exactly the rows the predecessor's commit
keeps. Attention reads the domain's retained rows from the query's window start; the window is
applied by the host-built visible spans, never by masking inside split-K.

The store compacts a history before a launch that would exceed a domain's span bound,
`ceil(row limit / rows per slab) + 16`, where the row limit is the context limit for Token and
`n` plus the largest advance (at most the context limit) for Window(n). Compaction moves one
domain's rows; kernels and batching follow the largest span bound of the model.

Each history slab contains every component of its domain at a fixed aligned offset. Dense
components hold activation values; affine components hold codes and scale/zero coefficients. A row
has the same slab index and offset in every component of its domain, so placement, compaction and
conversion move all parts of that row together. A slab targets 64 MiB: its row count is rounded
down to a multiple of the 256-row tile, with at least one tile. Bank slabs hold as many complete
banks as fit in that target, with at least one bank.

Elastic state growth is admitted before an advance. The store claims each required slab through
the heap before adding it, one slab per domain as needed, without replacing or copying existing
slabs. A refused claim or slab allocation leaves accepted numerical state, published backing, and
committed charge unchanged; growth that needs history slabs in several domains and bank slabs is
one fallible operation. The refusal returns an
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
An independently captured drafter boundary is tentative until the target and attached head
advance reconcile. Abort preserves the prior boundary; successful reconciliation publishes the
new row with the numerical state. Resume states own the row across forks and retention, while
generation holds only an obligation to use it while resident.
A started round is a value that owns its generation until its reconciled transition commits;
eviction and cancellation consume it and return the rewound or terminal generation.
A finishing prompt may publish its reconciled target token while its separately submitted
drafter-history entry remains in flight. That entry retains the binding right and all of its
leases until its own completion, then commits only if the target chunk was accepted. No
checkpoint, eviction, or dependent head submission can observe the unresolved pair of lanes.
Its physical service remains prompt work even when the token publishes before it completes.
A draft head that keeps no history (`HeadHistory::Target`) reserves no rows or banks: its head
reservation moves each request's accepted target state into the flight, which binds that state's
history planes read-only and returns it unchanged at physical completion or refusal. While held,
the request has no other target work, checkpoint or close, exactly as for an in-flight advance, and
reconciling the head's decision commits nothing.

**Binding right.** Each store has exactly one right to change its slab bindings: adding slabs,
moving rows and banks, and releasing slabs. The executor holds the stores' rights together with
the lookahead as one binding right (`StateBindings`). Growth, provisioning, residency, shrink,
compaction and tail relocation require it; row and bank claims (advances, checkpoints, forks, and
dropping claims) do not. Submitting a group moves the binding right into its flight and physical
completion returns it, so no binding changes while a flight is in the air; releases that need a
binding change run at completion. The lookahead, the step queued behind the last flight with its
drafter priming, is the only work that holds state between flights and lives in the binding
right. Its tentative rows are indistinguishable from another history's in every layout question
(a blocked last page, rows in place, growth), so it lives only until the next operation that
uses the binding right: a matching submission claims it, and every other such operation orphans
it on entry, before asking any layout question. A lookahead that embeds a device-selected row of
a streamed table ([residency](residency.md)) is queued before the host knows whether the device
holds that row. One whose row the device did not hold is not the continuation: no submission
claims it, it is orphaned, and the step is submitted again with the row brought. Growth has no
blocked outcome: it succeeds or
returns a memory deficit. Reservation recomputes its row requirements after
provisioning relocates shared tails, so availability is checked against the new
layout rather than charging again for the page relocation already consumed.

An interior accepted prefix with recurrent state requires numerical repair before the successor
can be published or checkpointed. State compaction, copying, and codec conversion follow the same
submit, complete, finish, reconcile lifecycle. Compaction requires the binding right, so it runs
only between flights after the lookahead is resolved. It moves rows and banks into free space of
slabs already held after submitted writes complete, and publishes the rewritten histories and bank placement only after every copy succeeds.
It merges a domain's spans before that domain's span bound is exceeded and empties the least
occupied slabs of every domain under pressure. Accepted state keeps its logical identity throughout. Cancellation, submission failure,
device failure and teardown release reservations through ownership.
Preemption releases physical request state while preserving accepted logical tokens. Restoration
replays only to the numerical position that existed before eviction. An accepted successor that
has not yet been consumed numerically remains the input for the next ordinary decode; replay must
not consume it early or advance beyond that numerical boundary.

Codec conversion reserves destination history before submission. One transaction owns both
stores' sequence claims, their source and destination slabs, both recurrent banks, and a
per-layer key/value mapping between codec components, each carrying its domain's rows. The mapping names the source and
destination codecs and row addresses; it is validated against a fresh destination, compatible
layer widths and recurrent layout, and the selected stores. Abort returns both unchanged states.
Commit publishes the destination position and history only after the state program finishes.

## Acceptance criteria

- No in-flight state transaction borrows sequence storage.
- Interleaved advances of concurrent sequences add no history span while the following rows in
  the same slab are free; a span never crosses a slab boundary.
- Every history fits each domain's span bound, including at full context and after a freed
  slab index is reused.
- One advance reserves rows in every stored domain or in none.
- A Window(n) history references exactly its last `n` accepted rows plus its tentative rows, also
  when an advance is longer than the window and when its rows cross slab edges; a checkpoint at
  `p` keeps rows `[p − n, p)` for every fork and resumed request.
- A store of one Token domain has one row numbering and one history slab tensor for every
  attention layer.
- A Block(rate) history holds exactly the blocks complete at its position: a commit of an accepted
  prefix keeps the blocks that prefix completes, a checkpoint at `p` references blocks
  `[0, p/rate)`, and forks and successors continue them.
- Compaction publishes only after all copies complete, preserves greedy continuation, and needs
  no new memory claim on any backend.
- Failed joint history and bank slab growth restores both published backings and their charge.
- A new sequence observes zero recurrent state even after prior sequences have returned dirty banks.
- The n-gram token window of version (bank, tape rows) and the host's tokens before that position
  name the same rows, across chunks, verification, rollback, forks and prefix reuse.
- A successor bank is never the zero seed, an accepted bank a live state, checkpoint or fork can
  read, or another in-flight successor.
- Submit failure and cancellation leave accepted state unchanged and release tentative claims.
- Full, partial, and zero acceptance reconcile each transaction exactly once.
- A recurrent interior prefix is not visible until its repair work completes.
- An accepted/resident checkpoint is created only after reconciliation.
- Exactly one binding right exists; every binding change holds it, and no binding change is
  possible while a flight holds it.
- Every operation using the binding right, other than the submission that claims a queued
  lookahead, orphans it on entry; no layout question is answered while it is queued.
- Growth succeeds or returns a memory deficit; it never silently does nothing.
