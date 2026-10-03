---
applies_to:
  - inference/engine/executor/src/placement.rs
  - inference/engine/executor/src/pipeline/**
  - inference/engine/executor/src/programs/native_target_graph.rs
  - inference/engine/executor/src/programs/native_target/pipeline.rs
  - inference/engine/executor/src/lanes/target.rs
  - inference/engine/executor/src/planning/resources.rs
  - inference/engine/executor/src/resources.rs
  - inference/engine/executor/src/domain/**
  - inference/engine/src/execution.rs
  - inference/engine/src/execution/**
  - inference/engine/src/worker/**
  - inference/engine/cli/**
---

# Explicit pipeline execution

The [review and qualification note](../../../inference/engine/cli/PIPELINE-REVIEW.md)
records evidence provenance and reproduction separately from this contract.

Placement intent is backend-neutral metadata over one original model. It names
ordered execution groups, each containing device-assigned partitions. A region
can identify original decoder blocks or a semantic tensor role, decoded axis and
half-open element range. Multiple tensor partitions in one group leave room for
TP; ordered groups of shards leave room for PP+TP. These descriptors authorize
no execution. The current pipeline projection refuses empty or sharded groups
before opening devices; the concrete executor additionally admits only two CUDA
whole-block stages. Tensor geometry, shard coverage, state/activation partitioning
and collective semantics require future admission, not an empty strategy label.

```text
Seismic DeviceTopology (selectors, physical memory pools)
             + ModelPlacement (groups, assigned regions)
                         |
                  pipeline projection
                         |
            two-stage CUDA capability admission
                         |
       local preparation/tuning -> ordinary Owner/worker/API
```

Seismic remains the sole topology and allocation identity authority. CUDA and
Vulkan views of one physical GPU share its memory ledger. No second topology or
interconnect measurement subsystem is installed. A future topology link must
separate discovered capability and measured bandwidth/latency from the executor's
selected transfer path; unknown properties stay unknown. This executor selects
bounded host staging regardless of peer access and records handoff durations as
qualification diagnostics, not reusable topology measurements.

A pipeline is an explicitly supplied ordered partition of one admitted original
model. Stages borrow nonempty contiguous ranges of that decoder, together covering
every block exactly once. Original global block identity names weights and history;
stage-local ordinal names traversal, and recurrent component offsets follow the
selected original blocks. No reconstructed decoder renumbers semantic identity.
The semantic model and logical state transaction are N-stage-shaped. The experimental
physical strategy is restricted to exactly two distinct CUDA execution devices,
plain text, one active request, one/two-row classes, no drafting, conditioning,
lookahead, retained-prefix or resume. Unsupported profiles are refused, not rerouted.

Each stage owns a separate allocation domain, heap, prepared programs, resident
weights, state store, graph workspace and output leases. Preparation and tuning
remain Magnitude's ordinary device-local work. Explicit paired construction repeats
the normal preparation on each supplied selector. Its one-request graph contract
retains two-row prefill without the ordinary multi-request slot allowance; ordinary
planning is unchanged. Each stage retains its local plan and
memory-pool identity. The ordinary throwaway warm-up runs through that same paired
domain before readiness; it accepts neither stage's request state. Local plan and
claim refusals retain their typed capacity cause and actual memory domain.
Startup preparation transfers the
existing heap, pools and unique store binding right into the loaded Owner without
reopening a device or recreating allocations. Pool adoption checks the exact
workspace allocation identity as well as local device, domain and admitted charge.
Fresh request admission checks and provisions both local stores before making
one request resident. Refusal publishes neither source; successful physical backing
growth on a refused admission remains measured headroom, not accepted state.
Only one installed request is allowed, including while its sources are in flight.
The ordinary Owner rejects requested retention and non-disabled prefix-cache ownership
in this profile; it does not silently ignore them. Resume and media are refused. Closing or evicting an idle request drops both
sources, and binding-right-controlled idle release covers both stores.
Both stages' unique binding rights belong to the same ordinary request domain;
its prefix allocation owner is not another request lifecycle. Physical holdings
and censuses remain separately classified per device, including current state
backing rather than a frozen startup estimate. Both devices are observed without
pooling their ceilings. The normal worker closes the concrete paired family behind
its existing generic execution owner and uses the same session/protocol. Additional
local reconciliations remain independently classified; observed allocation pool
identities, not vector order or the primary device name, label memory headroom. The numerical executor retains only
its immutable local resources and store identity, not binding-change authority.
Prepared programs or allocations
from one opened device never become another device's resources by matching shape,
backend or model name. Each stage's graph uses the normal parameterized decoder
builder and Seismic's certified layouts, projected against its local state plan.

Only assigned original semantic weight roles are imported, directly into the
owning residency store. The first stage owns embedding; the last owns final
normalization and vocabulary projection. Tied weights used on distinct devices
require distinct physical copies, charged independently. Converted resident
storage, constants, local state, graph pools and import peaks remain within each
local heap's fitting claims. Admission uses only that stage's assigned storage,
not a full-model plan with an enlarged capacity. Serial local graph families share
one workspace maximum; their output/upload pools remain independently charged.
The preallocated consumer activation buffer is a separate local transfer charge,
not model storage or implicit free scratch. VRAM is never pooled.

The boundary carries completed activation and required logical row metadata, not
persistent model weights, KV history or recurrent banks. PCIe host staging is
bounded and validates representation, exact geometry and destination ownership;
producer completion precedes transfer and consumer completion precedes release.
It promises neither P2P nor NVLink optimization.

Target reservation preflights both local stores and all required graph leases
before moving either source. Submission must preserve the reserved request identity,
row extent and accepted position. The same ordinary row constructor uses each stage's
own banks and history; only the final stage carries readout demands. Checked-launch
refusal before physical submission restores both sources. A no-readout prompt
chunk still completes both stages and accepts jointly, without inventing a token.
Logical acceptance waits for all physical stages and handoff to succeed, then
uses ordered all-stage preflight before any publication. A finished target result
retains the prefix advance alongside the suffix advance through generation
preparation. Joint reconciliation publishes both, or returns unchanged logical
sources on refusal; completed cancellation aborts both. Neither head nor stateless
results may publish pipeline state. Physical writes are not
rolled back by releasing reservations. A partial physical failure discards the
request. Synchronous paired submission yields to the existing control mailbox between
bounded physical steps, even with ample publication credit. Cancellation/drain retains outstanding state rights, leases and transfer
buffers until their device work completes. The existing Owner and public inference
lifecycle remain the sole owner lifecycle; no parallel demonstration server is an
execution strategy.

This strategy is never selected by fit or catalog policy. Explicit execution
validates the caller's partition; placement search, automatic selection and catalog
presentation remain outside this implementation.
A separate development-only fault-injection build can refuse suffix submission
after completed prefix execution and activation transfer, at an explicitly supplied
logical position. It uses the normal fatal/discard lifecycle, never simulates GPU
rollback or enters an ordinary experimental build. Numerical and lifecycle hardware qualification is separate from structural
metadata checks; a passing projection or ownership test proves no model generation.
