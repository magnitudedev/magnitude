# Principles

**The engine executes Magnitude's supported model configurations as efficiently as possible on consumer hardware, with dynamic memory use, effective handling of mixed workloads, and a path toward running larger models.**

## Scope

Magnitude's service owns model acquisition, installations, the catalog, and valid
configurations. The engine loads and executes compatible installed configurations
from the supported architectures and formats, including their encoders and
generation components such as drafters.

These principles guide architecture and tradeoffs. They establish objectives,
not a complete specification of scheduling, memory pressure, or execution policy.

## 1. Leading single-session performance

Aim for the fastest single-session prefill and decode compared with other engines
for the models and consumer hardware backends we support.

This objective concerns the complete inference path: model computation, kernels,
compilation, host coordination, and generation strategies, including speculative
decoding. Ordinary token-by-token decoding is not the limit of the performance
design. Prefill and decode each matter; strength in one does not establish success
in the other.

Single-session performance is a primary target, while real mixed-workload
efficiency remains an objective in its own right.

## 2. Dynamic, predictable memory behavior

Memory allocation is fully dynamic and follows actual inference demand as
histories grow, concurrent work changes, and reusable state accumulates. Shared
prefixes share storage, and resources that are no longer needed can be reclaimed.

Context-window and concurrency policies sit above the memory layer. They govern
permitted work and may change dynamically; they do not determine allocation sizes
or reserve state capacity. Memory is acquired for the work and state actually
needed.

Memory pressure is a normal condition on a user's system. Other applications may
compete for capacity, and inference demand may change over time. The engine must
respond deliberately and understandably when memory becomes tight. The specific
reclamation, waiting, and recovery policies remain design decisions.

## 3. Efficient, fair handling of mixed workloads

The normal workload includes concurrent conversations, continuations, forks,
session switching, short auxiliary requests such as chat titles, and large new
prefills. The engine should handle these patterns efficiently together, rather
than assuming an isolated session represents all usage.

Two principles drive scheduling and batching:

- **Opportunistic efficiency.** Exploit resources already loaded and computation
  already prepared, including reusable prefix state, resident weights, prepared
  inputs, and profitable batching opportunities.
- **Eventual service.** Requests are not indefinitely deferred, even when other
  work repeatedly offers better immediate efficiency. Requests needing more
  preparation still get their turn as readily executable work keeps arriving.

For example, a request continuing a history whose KV state is already available
can proceed directly with decode, while a new request may require a large prefill.
The scheduler should exploit the ready decode work while making room for the
prefill to advance. A continuing supply of convenient work must not prevent other
requests from receiving service.

## 4. Enable larger models

Use the available storage, memory, and device hierarchy to make models usable
where full residency would otherwise prevent them from running.

| Capability | Purpose |
|---|---|
| Expert streaming | Keep selected experts resident and bring others into execution memory as needed, with preparation ahead of use where possible |
| Embedding streaming | Load needed portions of embedding tables with minimal coordination overhead, including disproportionately large ordinary tables and large n-gram tables |
| Distributed inference | Eventually use connected devices together when their combined resources make a useful model configuration possible |

Placement and movement depend on the actual topology. Expert streaming may move
weights from disk into unified memory, or from system RAM into GPU memory. The
design should reason about these resource relationships rather than assume one
hardware arrangement.

Distributed inference is a longer-term target, not a current design priority.
The eventual objective includes discovering and characterizing devices and their
interconnects, then automatically choosing and tuning useful execution
configurations with understood performance. Interconnects may range from direct
high-speed links to Ethernet or suitable wireless networks. Detailed distributed
mechanisms need not be introduced before that work is in scope.

## Applying the principles

Evaluate a design by its contribution to these objectives and the costs it imposes
on the others. Scheduling, memory residency, and execution efficiency interact:
available state and weights influence the cost of work, while fairness prevents
those opportunities from indefinitely displacing other requests.

Demonstrate performance on both isolated sessions and representative mixed
workloads. Keep concrete mechanisms, acceptance criteria, and evidence distinct
from these overarching objectives.

## Principles by subsystem

| Subsystem | How the principles apply |
|---|---|
| Configuration and loading | Turn supported installed artifacts and component choices into efficient execution on the available hardware. Prepare and share resources deliberately, support partial residency, and keep loading independent of hypothetical context or concurrency allocations. |
| Model architectures and inputs | Express decoder, encoder, and drafter computation in a way that permits efficient execution across supported backends. Preserve the information needed for batching, reusable input preparation, and selective weight access. |
| Tensor compilation and kernels | Pursue leading prefill and decode performance across supported backends and workload shapes. Minimize coordination, data movement, and temporary memory costs while retaining efficient execution for mixed workloads. |
| Generation and speculation | Increase useful accepted-token throughput, accounting for drafting, verification, temporary state, and contention with other work. Speculation serves overall inference efficiency. |
| Scheduling and batching | Exploit loaded resources, prepared state, and profitable combinations of work. Balance these opportunities with eventual service: requests requiring more preparation are not indefinitely deferred by more immediately efficient work. |
| State and prefix reuse | Grow storage with actual histories, share common prefixes and forks, and reuse prepared state across requests. Retention should save computation while remaining responsive to memory pressure. |
| Memory and resource lifetime | Allocate for actual demand, account for live and temporarily retained resources, and reclaim safely. Context and concurrency policies remain separate from allocation; pressure produces deliberate behavior. |
| Weight residency and streaming | Make larger models usable by selectively retaining and preparing experts and embedding rows. Choose movement through disk, RAM, and device memory according to topology, and account for its effect on execution and scheduling. |
| Device execution and topology | Keep submission and completion efficient and expose the resource relationships needed for placement and movement. Distributed execution remains a longer-term extension of this concern. |
| Serving and request lifecycle | Carry diverse agent workloads into the engine with low overhead and clear progress, cancellation, and backpressure behavior. Expose enough state for Magnitude to explain delays and resource pressure. |
| Development, testing, and benchmarking | Make execution choices and resource behavior inspectable. Establish correctness and reproducible performance for single sessions, mixed traffic, dynamic memory, and streaming configurations through the production paths. |
