# Addressed Projection State

Addressed projection state lets projections keep small ordinary indexes in
memory and snapshots while moving large collection bodies into addressable
entries. The event log remains the source of truth; addressed entries are the
durable backing store for large projection-owned values.

## Core Boundary

Ordinary projection state owns collection indexes. These indexes describe the
logical shape of a collection and point at addressed entries.

The address-space runtime owns residency: resident values, dirty flags, pins,
flush-on-release, and changed-address notification.

The entry store owns durable encoded entries. It loads and flushes values by
namespace and address. It does not know projections, events, pins, display
views, or collection semantics.

Addressed collections sit between projection logic and raw addresses. They
translate semantic collection operations into addressed reads, writes, windows,
and invalidation.

## Sequence Identity

Logical sequence segments are not physical storage entries.

An addressed sequence index contains logical segments such as `seg-0` and
`seg-1`. Each logical segment points at a physical addressed entry such as
`.../entries/entry-12`.

A sequence window carries:

- the logical segment id, for explanation and debugging;
- the physical entry address to read;
- offsets into that physical entry;
- expected item ids for the requested slice.

A window is valid against the physical entry address it captured. It must not
derive a storage address from the logical segment id.

## Physical Entry Reuse

A physical sequence entry may be reused only when every previously indexed item
keeps the same offset in the rewritten entry.

| Operation | Reuse old entry? | Reason |
| --- | --- | --- |
| Content-only update | Yes | Item ids and offsets are unchanged. |
| Tail append | Yes | Existing window offsets remain valid. |
| Insert before existing items | No | Existing item offsets move. |
| Delete or reorder | No | Existing item offsets or ids change. |
| Item id change | No | Existing windows validate item identities. |

Structural movement allocates a new physical entry. Old physical entries remain
readable for windows that already captured them.

This is not version hashing and not best-effort repair. The ordinary index is
the allocator state, and allocation is driven by the structural reuse rule.

## Validation

Addressed sequence reads validate the ordinary index or window against the
physical entry contents. The stored item ids must match the expected ids at the
expected offsets.

A mismatch means the framework's collection/index/address invariant was broken.
It is not a transient read failure and should not be handled with retries.

## Locator Discipline

The ordinary index is both allocator and locator. Projection handlers must
address every mutation through it — by item id or position — and a mutation's
cost must be proportional to the affected suffix, never to the collection's
total size.

Handlers never read a whole sequence to mutate part of it, and never locate an
item by scanning content. When a handler needs to find something, the locator
is either the index itself (item ids, tail suffixes, positions) or small
ordinary projection state maintained for exactly that purpose — for example, a
map from an external identity (a stream id, a child fork id) to the message ids
it owns. Such locator state is an invariant, not a hint: a miss is broken
integrity, not a cue to fall back to a scan. Counters and locator maps are
maintained incrementally by the mutations that change them, never recomputed by
reading collection contents.

Whole-sequence reads are reserved for operations that are genuinely
whole-sequence — restoring a collection wholesale, or materializing a
requested window — and must justify themselves at the call site. Helper
surfaces should not offer bulk read-transform-replace operations at all, so the
expensive path cannot be reached for out of convenience.

## Pins And Residency

Residency is exactly the pinned set. There is no cache policy, no capacity, and
no eviction schedule — residency is derived from the pins the system already
records:

- Reads of unpinned addresses return values without creating residency; every
  such read resolves through the entry store. Absence is never cached: a store
  miss is surfaced to the caller, and collections convert misses on indexed
  addresses into their integrity error.
- Pins fault their targets in and hold them resident. Pins are a subset of the
  resident set by construction, so an entry's pin count is authoritative.
  Because dirty entries are flushed before they are dropped, every
  index-referenced entry is resident or stored; a pin target that is neither is
  broken integrity — a defect, not a recoverable error.
- Writes dirty their entries. When an entry's last pin is released, it is
  flushed if dirty and then dropped. A commit that leaves an entry unpinned
  flushes it at that commit (write-through). Flushing is a precondition of
  dropping and of snapshot capture, never a schedule; durability between those
  points is the event log's job.

Pins claim physical addresses, not logical segment ids.

View pins are derived from accepted display-view windows. A view pins the
physical entries it needs to materialize its accepted shape.

Producer pins are owned by active writers such as streaming assistant messages,
thinking messages, tools, or worker communications. They are claims by actual
current work — never anticipated future use, which would be a cache heuristic
in disguise. If a structural rewrite moves a producer's target item to a new
physical entry, the projection must replace the producer pin with the new
physical address.

Pin replacement acquires the new addresses before releasing old ones, so a
moving view or producer does not briefly drop residency for its active target.

The derived I/O behavior: streaming writes to producer-pinned entries perform
no store I/O per update and flush once at release; unpinned writes flush once
at their commit, proportional to real events; replay I/O is bounded by
producer-pin spans, not by delta count.

## Snapshots And Persistence

Projection snapshots contain ordinary addressed indexes, not addressed entry
bodies. Dirty addressed entries are flushed in place before snapshot capture so
the snapshot's indexes can be resolved through the entry store.

Startup must restore ordinary indexes only when they are safe relative to the
event log. If a projection snapshot is stale or has an event-log suffix that
would replay against newer addressed entry files, startup must replay from the
event log instead of mixing old ordinary indexes with newer physical entries.

## Old Entry Lifecycle

Old physical entries are intentionally retained after structural rewrites so
captured windows can remain valid.

Deleting unreachable physical entries is a separate compaction or garbage
collection concern. It must be explicit and must account for open views,
snapshots, and any other readers that can still hold windows. Ordinary
collection mutation must not delete old entries as a side effect.
