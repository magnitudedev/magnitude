# Structure and accounting

Authored structure exposes the work an implementation must perform and the dependencies it must respect. Physical construction turns that structure into actual resource demand. Evaluation compares that demand with target supply or measures its execution.

```text
source operations, domains, values and effects
                  |
                  v
logical work, multiplicity, dependencies, available parallelism
                  |
          physical construction
                  v
actual instructions, accesses, storage, communication and overlap
                  |
       +----------+----------+
       |                     |
target supply model      native execution
       |                     |
       v                     v
estimated cost           measured cost
```

## What source provides

Shapes and iteration domains determine logical work multiplicity. Value uses and carries determine dependencies and liveness. Views and indexing expose access geometry. Parallel regions expose independent work; ordered regions expose recurrences. Effects constrain legal reordering, sharing and reuse.

These facts are derived from the same checked operations that define execution. A separate author-maintained operation count would duplicate authority and become stale when a body changes.

Source facts are not yet hardware costs. A logical tensor read might be served from a register, shared storage, cache or device memory. A computed intermediate might require no allocation. Several source operations might map to a permitted compound instruction. Indexing, packet decoding and synchronization may introduce real physical work.

## What physical construction adds

The selected physical structure determines participant mapping, instruction forms, data movement, storage lifetimes, communication, synchronization and launch/control overhead. It owns both these actions and the resource projections derived from them.

An accounting pass can summarize that structure; it cannot independently choose another layout or omit supporting work because it did not appear explicitly in source. Resource reservations and performance estimates may use different projections of the same structure, but neither supplies a competing executable description.

## Demand, supply and cost

| Quantity | Meaning |
| --- | --- |
| Exact count | A fact derived from a fixed execution path and its operands. |
| Capacity bound | A safe envelope over paths or runtime values; sufficient for reservation, not necessarily the actual count. |
| Target limit | A hard legality restriction such as a native allocation or dispatch limit. |
| Target service model | Throughput, latency, occupancy and contention behavior used for prediction. |
| Cost estimate | A prediction combining physical demand, dependencies and target service behavior. |
| Measurement | An observation of a concrete executable, invocation and environment. |

Supply and demand enable pruning of impossible assignments, identification of bottlenecks, and comparison of legal candidates. Hard limits never become soft penalties. A weak cost model never makes a numerically or structurally invalid candidate usable.

Branch probabilities and workload frequencies must be explicit assumptions or observations. A preparation range constrains where to spend search effort; it does not imply that values in that range are uniformly likely.

The compiler's [analytical evaluator](../compiler/analytical-evaluation.md) owns prediction, while the [feedback evaluator](../compiler/feedback-evaluation.md) owns controlled measurements. Runtime [resource ownership](../execution/resources-and-lifetimes.md) concerns actual capacity and lifetimes.

