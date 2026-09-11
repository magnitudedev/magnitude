# KV state

**Physical pages are claimed by extents; logical visibility belongs to the
sequence. A read is a stable window plus explicit visibility, so histories grow,
fork and fragment without a kernel, a peer or a copy noticing.**

## Structure

```text
pool ── slabs (one device allocation each, pages of fixed token width)
          └── extents: a contiguous page range, a generation, a claim count
                └── runs: one claim each; the same extent under several owners
                      └── spans: run + logical start + visible length      (a sequence's history)

read run:   consecutive spans, physically adjacent, fully used   → one contiguous operand
read group: read runs in one slab                                 → one window + segment metadata
```

The physical side knows pages and claims; the logical side knows positions and
lengths. Nothing joins them except a span, and a span is the sequence's.

## Rules

| Rule | Reason |
|---|---|
| A claim keeps an extent; the last claim frees it | Checkpoints and forks share extents without copying, and nobody tracks who else holds them |
| An append writes in place only into an exclusively claimed tail | A shared tail belongs to a checkpoint too; writing into it would rewrite someone's past |
| Growth prefers the sequence's own frontier | Adjacent extents form one read run, one operand, one launch |
| A growth preference never claims pages | Anticipating a long prompt sizes the slab; it does not take capacity from peers |
| A read window is stable while extents grow | The operand spans from the earliest extent to the slab's end, so growth changes metadata, never the specialization |
| Metadata grants visibility; the window never does | A window may cover gaps and future pages; only listed segments are read |
| Fragmentation changes addresses, never the reduction | Attention's summary over a history is one logical reduction whatever runs it is split across |
| A read pins its extents through completion | The sequence may commit, fork or evict while the command is in flight |
| Reclaimable means whole slabs released exclusively | Freeing pages inside a slab reclaims nothing from the device |

## Capacity classes

A read is specialized on capacity, not length. The window's capacity is fixed by
the slab; the segment capacity is rounded to a power of two; the segment count is
small. Lengths are operands. A history that grows by one token each step never
compiles a new kernel, and a history spread across many extents costs a segment,
not a specialization.

## Forking

```text
A: [p0 p1 p2 p3]                         one extent, one claim
checkpoint A ──► C                       C claims the same extent; A's tail is now shared
A appends ──► [p0 p1 p2 p3][a4 a5]       A takes a new extent; C's view is unchanged
fork C ──► B                             B claims p0..p3; appends into its own extent
release C, then A                        p0..p3 survive while B holds them
```

A fork costs claims, never bytes. Sharing is invisible to every reader, because a
reader sees a window and segments, and the segments name only what that reader
may see.

## Reclamation

Pressure is priced by what a victim set would release exclusively: extents whose
every claim belongs to the set, and only whole slabs. Closing the pool retires no
live extent; the last claim on each frees it, whenever that is. Nothing here
decides who is evicted; that is service policy over these prices.
