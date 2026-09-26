# KV state

**Magnitude owns logical history, sharing and acceptance; Ops owns the
physical tensor resources and ordered reads and writes that realize each
tentative state version.**

## Structure

```text
Magnitude logical state
└── sequence spans: logical start, visible length, ownership
    └── runs: adjacency and sharing of resource slices
        └── Ops resources: physical storage and completion lifetime

tensor invocation
├── read metadata: which logical segments are visible
├── write metadata: destination slices for the tentative append
└── versioned resources: append result consumed by subsequent attention
```

The logical side knows positions, prefix identity, checkpoints, forks and
acceptance. The tensor side knows resource identity, views, access ranges,
dependencies and physical completion. A stable state view is the contract between
them; neither owner adopts the other's policy.

## Rules

| Rule | Reason |
|---|---|
| A logical claim retains its Ops resource slice | Shared prefixes survive without copying and physical ownership remains explicit |
| Append uses an exclusive tail or claims a new run | Tentative work never rewrites history visible to a checkpoint or fork |
| Visibility is explicit metadata | A physical window may contain gaps or future capacity without making either readable |
| Resource writes produce a new graph version | Attention cannot read before append or observe unordered mutation |
| Logical publication waits for completion and acceptance | Finished device work may still be rejected; accepted work must already exist |
| Reads retain their resources through completion | Commit, fork, cancellation and eviction cannot invalidate in-flight tensors |
| Fragmentation changes resource views, not attention mathematics | Storage policy does not create a second attention operation |

## Capacity and specialization

Page size and extent placement are Magnitude state policy. Ops receives
bounded resource views and visibility metadata. Stable capacity classes keep
continuously changing history length dynamic while bounding compiled attention
specializations.

Growth prefers physical adjacency because fewer resource segments reduce binding
and traversal cost, but the state remains correct when fragmented. The compiler
may choose streaming, partitioned or materialized attention from the actual
geometry without learning why the runs were placed that way.

## Forking and checkpoints

```text
accepted prefix ──► checkpoint shares logical claims and resource slices
      │
      └── next tentative append uses a private tail or a new run
```

A fork copies ownership descriptions, not tensor bytes. Each descendant receives
its own subsequent resource version. A checkpoint is created only from reconciled
state; no unresolved execution or tentative version is published as history.

## Reclamation

Magnitude prices which logical owners could release a set of extents. Ops
reports the physical bytes and outstanding executions retaining their resources.
Capacity is reclaimed only when the final logical claim and every submitted use
have both ended.

The service chooses eviction. State placement does not select kernels, and
Ops does not select victims or infer logical visibility from allocation.
