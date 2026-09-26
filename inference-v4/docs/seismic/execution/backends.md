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
| Metal | Pipeline and resource bindings, command encoding/submission, residency where required, synchronization and command completion. |
| CUDA | Module/function bindings, streams/events, dispatch and synchronization, and any selected native allocation/mapping lifecycle. |

A facility enters the compiler's domain through its complete typed [target contract](../compiler/target-contracts.md), including construction, emission, binding, resource and numerical handling.

## Ordering and completion

The physical/workflow plan specifies the required happens-before relationships. Backend issue preserves them through its actual execution mechanisms. A successful enqueue establishes submission, not result availability.

Cross-queue or cross-stream use needs the represented dependency before the consumer accesses the resource. Planned releases occur after every use. Participant barriers and collectives use their declared topology and progress assumptions; the runtime cannot assume arbitrary groups are simultaneously resident.

Completion publishes actual result words and storage. Integer, natural/index and range values remain typed through native readback. Failure status is initialized and published by its represented protocol, including empty execution paths.

## Errors and teardown

Malformed user arguments fail earlier in binding. Actual device loss, native allocation failure, submission error and service shutdown are backend outcomes. A malformed admitted executable or a violated fixed target contract is a compiler/backend defect rather than a trigger for fallback selection.

After partial submission, the backend preserves all potentially live resources until terminal completion or the native service's established terminal failure state. Caller cancellation or handle drop cannot release them early. [Resource ownership](resources-and-lifetimes.md) defines this lifecycle across backends.

## Direct native route

Explicit authored native execution binds the portable interface to the selected asset and launch declaration. It shares the requirements for actual device identity, argument rights, native resource ownership and completion. It does not run the candidate evaluator or acquire the prepared pipeline's numerical guarantee.

Direct native failure is reported directly. It cannot silently switch to a compiler candidate, just as the compiler pipeline cannot silently switch to an interpreter or authored native asset when construction is incomplete.
