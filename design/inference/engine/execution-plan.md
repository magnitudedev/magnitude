---
applies_to:
  - inference-v4/engine/model-executor/**
  - inference-v4/engine/model-kernels/**
  - inference-v4/engine/model-state/**
  - inference-v4/engine/src/execution.rs
---

# Numerical execution plan

Before allocating persistent device memory, a device-aware planner resolves one immutable plan
for enabled components, exact weight representations and tied storage, ordered checked-entry
program slots, method capabilities, service policy, and all device resources. The ordered program
topology is both the complete requirement set and the construction recipe. Program construction
attests every exact slot and returns typed callable groups; execution never queries a handle map or
coarse coverage class after readiness.
The composition root prepares complete Seismic workflows for the admitted model geometry and
finite launch classes, imports the target component, allocates the storage reported by those
workflows, and publishes readiness only after those steps succeed. The engine does not maintain a
second numerical tensor-shape description.

The resource plan authorizes persistent weights and state, including the permanently pristine
recurrent zero seed, concurrent typed workspaces, outputs
that outlive workspaces, variable retention capacities, optional component residency, and the
qualification/startup peak. Persistent allocation follows planning. Qualification scratch is
released before readiness. Execution receives plan-issued leases and cannot allocate general
scratch outside the plan.
Variable retained feature tensors are allocated by ResourceAllocator only after an exact byte
charge against the planned retention limit; their ownership claim refunds that charge on drop.

Seismic composes native checked entries into prepared workflows for decoder blocks and other
numerical units. Its checked entry contracts derive graph-local mutable scratch, host-uploaded
inputs, intermediate and result tensors' representations, extents, alias conditions, and lifetimes.
It reports exact storage charges and owns bounded concurrent execution slots. Compatible launch
classes share one physical scratch arena per concurrent slot, charged at their maximum footprint.
Resident imports use one-shot destinations and startup upload storage derived from their checked
contracts. The engine charges Seismic's reported native invocation, intermediate, and result
storage; it does not author parallel tensor recipes or look up named intermediates during a
request.

The target and head workflows cover the admitted row and history-segment ladders. Decoder numerical
pipelines share projection results through checked Seismic result edges. Dense feed-forward owns
its activation product; attention owns normalized input, Q/K/V projections, prepared rotary state,
stable-softmax accumulation, and gated values; routed feed-forward owns normalized input, router
logits, selected expert products, shared products, and the shared coefficient. Seismic prepares an
exact workflow for the selected physical batch class, so a small decode batch does not execute the
maximum class width. Linear projection stages use cooperative subgroup reductions. A monolithic entry that recomputes normalization,
projection, routing, or softmax for each output coordinate is not an admissible production program.
Target readout preserves every demanded feature row, then gathers only rows that demand logits
before vocabulary projection. The projected-row capacity follows the service's finite decode and
request-batch bounds; prefill row capacity does not imply the same number of logits rows. The
gather and projection remain inside one checked Seismic workflow with one owned output lifetime.
Conditioning overlays are exact request-dependent Seismic workflows. Before the embedding entry
runs, construction binds their source spans and mutable destination views to the reserved embedding
output and seals every checked call. These external-only overlays add no scratch or result storage;
the embedding result remains under its original output lease.
History publication occurs only after attention has finished reading all visible and fresh spans.

Vision patch capacity is the admitted merged output row limit times the merge area; input validation
rejects a larger aggregate before reserving a vision slot. Vision attention sees every physical
patch row, so its prepared workflow uses the exact admitted patch-row count; padding with additional
patches would alter real outputs. Seismic's recurrent workflow derives
the exact window and delta tensor contracts from its checked entries. Each recurrent block binds
the accepted and successor state for its exact active request slots; checked gather and scatter
operations share the block's ordered submission with its numerical computation. The engine does
not dispatch separate state transfers around the block or bind padded request state. Workflow slots cover
submitted concurrency, and retained outputs cover submitted and live request owners. Source-weight
upload uses the largest admitted encoded tensor as a one-shot startup resource. Qualification and
upload are sequential, so the startup-only addition is their maximum. Prepared native argument,
intermediate, and result storage is charged from the prepared workflows and checked against their
actual allocations.

One program factory is selected at startup. Native execution may return an already ready
submission, while compiler-planned work may remain pending. Both submissions own their validated
launch, state transactions, workspace, and output until completion. Finishing physical work
returns the output and the launch's reconciliation payload. A failure or drop releases the same
owned resources through RAII.
Before submission, the service resolves exact typed requirements against read-only availability,
applies retention eviction or live preemption while selection remains provisional, and takes one
owned reservation. That reservation contains every workspace, output, and state successor claim;
the service also holds exact publication-ring permits for the result. Launch construction consumes
those claims and performs no allocation. A capacity result after reservation is an invariant
failure. Device and invariant submission failures terminate the domain. Recurrent repair retains
its exact accepted prefix and successor claims through this ownership path.

The worker constructs one concrete program family for target, head, projection, vision, and state
maintenance. The executor domain and service owner are generic over that family, so each flight
keeps its concrete submission type through completion and reconciliation. The root seals the
generic owner behind its non-generic worker command interface before exposing the engine client.
Replacing the family changes program construction without changing admission, scheduling,
generation transitions, or publication.

## Acceptance criteria

- Every required entry corresponds to one ordered, attested callable slot.
- No independent kernel requirement set, semantic class set, or runtime handle query remains.
- No independent numerical tensor recipe or request-time role lookup remains; Seismic owns each
  prepared workflow's complete tensor contracts and reports its exact storage charge.
- Recurrent state movement and computation use one checked workflow for the exact active request slots.
- Every persistent, workspace, output, retention, and startup-peak byte traces to the plan.
- Native-ready and planned-pending submissions drive one executor lifecycle.
- The same service owner accepts either family without lane-specific path branches or erased submissions.
- Finished work retains all state needed for exactly one reconciliation.
