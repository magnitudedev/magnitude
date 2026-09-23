# Resources and lifetimes

The compiler derives the resource plan from actual execution. Runtime acquires and owns the corresponding native capacity and access rights. The plan and the resources have different responsibilities; neither is a certificate for the other.

| Input | Output |
| --- | --- |
| Complete workflow, invocation-resolved requirements and runtime capacity/services | Admitted run owning bindings, leases and planned lifecycle actions; then submitted ownership retained to completion |

## What the plan contains

The executable/workflow identifies external buffers, internal data storage, scratch, descriptor tables, ABI argument data, scalar publications, failure status and native command/context requirements. Each requirement has its actual use interval and access relation.

Direct and segmented storage use the compiler-selected ABI. Every page and descriptor level is included. Host-written argument or address tables are live data consumed by native commands; owning their allocation alone does not make it safe to overwrite their contents.

Resource expressions available at admission use only invocation-known values and fixed target/choice facts. Loop-local indices and execution-produced scalars cannot escape into them. The compiler closes sequential loops by maximum reusable demand, empty loops by zero and concurrent use by simultaneous capacity. Runtime evaluates those constructed expressions rather than re-deriving loop bounds.

For execution-dependent geometry, the plan reserves a source-derived capacity envelope or specifies a later native lifecycle action. It preserves the exact execution value separately. Failure to obtain aggregate capacity is legitimate; wrapping a size conversion or silently shrinking the logical value is not.

## Transactional admission

Admission acquires the capacity and access required for the whole workflow before submission. If acquisition fails, it releases provisional acquisitions and submits no partial workflow. Leases represent actual resource rights and prevent conflicting reuse.

Where the target requires allocation or mapping at a later native execution point, admission reserves the corresponding plan and capacity/access commitments. The executable already contains the acquire, publication, use and release dependencies. This permits planned lifecycle actions without introducing unplanned growth or runtime repair.

The native service can still fail an allocation or lose the device. Such external failures are reported through execution ownership, which retains anything already issued. Admission is not a promise that an external service cannot fail later.

## Reuse and persistence

| Resource | Reuse condition |
| --- | --- |
| Sequential scratch | Prior use has completed and the selected lifetime plan permits reuse. |
| Overlapping scratch | Distinct capacity or a real dependency eliminating overlap. |
| Loop carry/state | Retained across every iteration or invocation that uses it. |
| ABI/descriptor contents | Every prior native reader is complete before host overwrite. |
| Output storage | Publication is complete and consumer ownership permits reuse. |
| Pooled native allocation | No outstanding access lease or potentially live native use. |

A persistent allocation pool keeps native allocations between runs to avoid repeated allocation cost. It is not disk persistence, does not define logical program state, and does not relax lifetime rules. A cache entry retaining an allocation must not outlive the accounting owner that charges it without transferring that ownership.

## Submission, failure and completion

```text
admitted ownership
       |
       +-- failure before issue -> release acquisitions
       |
       v
some or all native commands issued
       |
       +-- success / cancellation / error / handle drop
       |       ownership remains while work may be live
       v
terminal completion established
       |
       v
publish outcome, release or return resources to pools
```

An error after partial issue does not rewind native execution. Retain all resources and argument rights that might still be in use. Cancellation stops further work where possible; it does not imply already issued work is complete. Completion handling must survive caller-handle drop.

Resource ownership composes through branches, loops, native queues/streams and workflow nodes. The same last-reader/last-writer relationships that permit overlap determine reuse. Backend completion signals implement those relationships; the host cannot infer them merely from successful enqueue.

See [physical IR](../compiler/physical-ir.md) for how the plan is constructed and [backends](backends.md) for native execution responsibilities.

