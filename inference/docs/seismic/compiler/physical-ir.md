# Physical IR

Physical IR represents what a candidate actually executes: computation, control, storage, communication, synchronization, publication and resource lifecycle. It is below checked language and above native emission.

| Input | Output |
| --- | --- |
| Source-directed construction with selected or symbolic physical choices | One coordinated execution representation; after closure, `ClosedExecutableIr` with derived resource requirements |

It is not a second description of a candidate beside an authoritative schedule. Kernels, launches and regions are components of this representation. Derived views may be cached under its owner, but they cannot be independently authored or remain valid after the underlying structure changes.

## What it must represent

| Component | Owned information |
| --- | --- |
| Values and operations | Actual producer/use bindings, arithmetic, representations and numerical modes. |
| Work assignment | Logical domains, participant mapping, partitioning and launch geometry. |
| Storage | Logical identities, direct/segmented realization, views, initialization, versions and access rights. |
| Control | Branches, loops/carries, dependencies and actual runtime values. |
| Communication | Transfers, visibility, barriers/events, pending results and completion. |
| Native execution context | Queue/stream/context, participation and behavior-affecting operation descriptors. |
| Publication | External results, writable-state updates, scalar transports and failure status. |
| Resources | Allocation and descriptor requirements, layouts, bindings, lifetimes and planned acquisition/release actions. |

Full physical-domain coverage requires the permitted native operation and composition surface. A narrow launch/repeat list cannot claim exhaustiveness over targets with additional async, queue, cluster or lifecycle choices. Progress is part of that contract: output arithmetic alone does not establish legal scheduling.

## Construction and closure

```text
source-directed regions
        |
        v
seal -> normalize -> derive lifetimes/layout/reuse -> close
                                                    |
                                                    v
                                           ClosedExecutableIr
```

These may be private consuming states of one construction. They are not independently owned pipeline layers. Normalization precedes derived resources and identity. A rewrite consumes the affected structure and rederives affected projections; a stale resource array cannot survive beside a rewritten executable.

Closure derives requirements from all actual operations, including launch, copy, fill, scalar transport and ABI uploads. Each closed launch owns its kernel, placement, storage uses and target ABI projection. Whole-executable closure folds these uses through control into valid reservation expressions. Launch-local closure can still refer to its lexical environment; invocation admission cannot. Callers cannot supply a parallel resource inventory.

Specialization substitutes actual choices and preserves definedness, source correspondence and numerical meaning. Imports remap owner-local storage/kernel/slot IDs together. Same-entry imports share the expression arena; foreign entry objects reconstruct through the receiving semantic owner.

## Logical storage

A logical allocation has one identity used by views, versions and hazards. Its physical realization is either direct or segmented. Representation planes and access rights remain above the resolver. Every backend, copy/fill, intrinsic and binder consumes the same selected storage ABI.

One semantic value/version may have several physical copies or residences. Their allocations and access paths can differ; an older replica cannot stand in for a new version after a write. Native resource identities and kinds are distinct from logical tensor identity. Resource requirements retain their actual unit, alignment, access and lifetime; events or participant capacity do not become byte buffers.

For native allocation limit `M`, choose a data page size aligned to the indivisible scalar/atomic units required by the allocation. Each page fits `M`; bounded-fanout descriptor tables fit too. Repeated descriptor levels cover all data pages. Zero-length storage follows the zero-size ABI and is never dereferenced.

Indivisible atomic words cannot cross page boundaries. Packed non-atomic fields may be assembled across pages by the common decoder. A compound native operation requiring contiguous storage receives a constructed contiguous tile; it cannot treat a segmented descriptor as a pointer.

Logical size and address arithmetic cannot assume a product of legal external extents fits a native word. Host capacity evaluation uses exact integers; device addressing uses enough fixed-width limbs or equivalent mixed-radix coordinates for the source-derived maximum. Conversion to a native length or address occurs only at a bounded constituent. Overflow never becomes a successful smaller allocation.

The ABI owns address width, descriptor layout, alignment, pointer encoding and residency requirements. All pages, table levels and publication actions participate in accounting and lifetimes. Aggregate capacity may legitimately be unavailable; a per-allocation limit cannot silently shrink the entry domain.
An internal reservation determined by invocation inputs and fixed choices must satisfy the target's per-allocation and index limits across the accepted domain, even if its backing is acquired at a reached schedule action. Execution-produced reached sizes retain their explicit capacity-failure path.

Public arguments/results preserve the external ABI. Internal carries and call results can be segmented; publication copies their logical values to the external destination when required.

## Scoped resources and lifecycle

Resources derive from actual use intervals. The ordinary device context is inherited from its region, whose sequence supplies device ordering. Host issue order does not imply device completion. Only consumed producer/completion dependencies require explicit references; ordinary sequencing needs neither a second edge list nor a native event per operation.

Reuse requires every conflicting previous native reader/writer to precede the new use under that execution relationship. An ordinal is a compact implementation only within a region whose ordering justifies it. Host-written ABI ranges remain live through the last native reader, so their overwrite needs completion or separately reserved versions. Existing backend completion ownership enforces these uses; no companion lifetime tracker is added.

Sequential scratch reuse takes maximum demand only where those completion relationships allow reuse; overlapping uses need simultaneous capacity. Execution-dependent geometry uses a safe invocation-time capacity envelope or a resource lifecycle already represented in execution. Bounds reserve capacity without replacing exact runtime values. Fixed internal upload or encoding protocols may remain inside an operation; selectable variations affecting work, cost, storage or overlap belong in its selected contract before closure.

The plan reserves required capacity/access before submission. It may contain native allocation, release or mapping actions at their specified execution points. Unplanned allocation growth, repair and candidate reselection during execution are forbidden. [Runtime ownership](../execution/resources-and-lifetimes.md) realizes the plan and retains resources through actual completion, including partial failure.
