# Backend execution

CPU, Metal and CUDA execute the same bound physical contract through different native mechanisms. Backend drivers own native issue and completion, not candidate selection or numerical qualification.

| Input | Output |
| --- | --- |
| Admitted bindings, executable/native artifacts and planned commands/resources | Submitted ownership, terminal completion and typed results or native failure |

## Common responsibilities

Each backend implements the selected argument/storage ABI, logical address resolution, copy/fill semantics, participant synchronization, result/status publication and resource lifecycle. Native handles and buffers remain owned through their last use.

No backend may silently change numerical modes, replace an unavailable operation or reinterpret a segmented descriptor as a direct pointer. Such choices belong in [physical construction](../compiler/physical-ir.md) and [native formation](../compiler/native-emission.md).

| Backend | Native execution concerns |
| --- | --- |
| CPU | One process-wide worker pool (a participant per physical performance core, the submitter included); a native submission runs as one pool job with a barrier between launches, each launch using at most as many participants as it has work items; host address bindings, dependency completion and the required saved/restored floating environment. |
| Metal | Pipeline and resource bindings, command encoding/submission (a submission's independent launches together in concurrent encoders, the encoders in order: see Ordering and completion), residency where required, synchronization and command completion. |
| CUDA | Module/function bindings, streams/events, dispatch and synchronization (a replayed submission's launches as a graph in stream order: see Ordering and completion), and any selected native allocation/mapping lifecycle. |

A facility enters the compiler's domain through its complete typed [target contract](../compiler/target-contracts.md), including construction, emission, binding, resource and numerical handling.

## Ordering and completion

The physical/workflow plan specifies the required happens-before relationships. Backend issue preserves them through its actual execution mechanisms. A successful enqueue establishes submission, not result availability.

Within a direct-native submission the launches' bindings are those relationships (`runtime/src/native/hazard.rs`). A launch reaches byte ranges of allocations: each bound view, and through a slab view's address table each backed slab region. It depends on an earlier launch when one of them writes bytes the other reaches. A host port (`&host mut tensor`) is written by the host once it sees the launch, so the launch is ordered as its writer although the device never writes it: a protocol the host serves while work runs, such as the expert watcher's requests and parks, is ordered by those declarations. Metal places each launch at its dependency level before encoding: one past the highest level of an earlier launch it depends on, so a launch runs as early as its dependencies allow, before earlier launches it does not depend on, never before one it does. No launch crosses a timeline operation. Each level runs in one concurrent encoder; consecutive levels of one launch share a serial encoder, which orders them at no cost beyond the dispatches themselves. Every direct encoder declares one device buffer written (`useResource:usage:`) that no launch binds; Metal's hazard tracking then runs the queue's direct encoders one after another. Per-launch timing still encodes each launch alone, in submission order. A CUDA replayed graph keeps stream order, a chain of its launches, and CUDA's launches outside a replay (standalone calls, a timed trace) run in stream order too.

Cross-queue or cross-stream use needs the represented dependency before the consumer accesses the resource. Planned releases occur after every use. Participant barriers and collectives use their declared topology and progress assumptions; the runtime cannot assume arbitrary groups are simultaneously resident.

Completion publishes actual result words and storage. Integer, natural/index and range values remain typed through native readback. Failure status is initialized and published by its represented protocol, including empty execution paths.

## Errors and teardown

Malformed user arguments fail earlier in binding. Actual device loss, native allocation failure, submission error and service shutdown are backend outcomes. A malformed admitted executable or a violated fixed target contract is a compiler/backend defect rather than a trigger for fallback selection.

After partial submission, the backend preserves all potentially live resources until terminal completion or the native service's established terminal failure state. Caller cancellation or handle drop cannot release them early. [Resource ownership](resources-and-lifetimes.md) defines this lifecycle across backends.

## Direct native route

Explicit authored native execution binds the portable interface to the selected asset and launch declaration. It shares the requirements for actual device identity, argument rights, native resource ownership and completion. It does not run the candidate evaluator or acquire the prepared pipeline's numerical guarantee.

Direct native failure is reported directly. It cannot silently switch to a compiler candidate, just as the compiler pipeline cannot silently switch to an interpreter or authored native asset when construction is incomplete.
