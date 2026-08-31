---
applies_to:
  - cli/src/commands/**
  - cli/src/index.tsx
  - cli/src/server/service.ts
  - cli/src/agent-docs/**
  - packages/client-common/src/harness-connections/**
  - cli/src/harness-connections/**
---

# Non-interactive CLI contract

## Scope

The non-interactive CLI is a human-readable, agent-usable projection of Magnitude product state.
It does not expose transport documents or internal state graphs. Bare `magnitude` and `magnitude
setup` remain interactive entrypoints and are outside this presentation contract.

The public command vocabulary is:

```text
update
service install | uninstall | start | stop | status
hardware
catalog list | show <model-id> | recommendations [--preference <value>] [--limit <count>]
catalog pull <model-id> | cancel <model-id> | remove <model-id>
models status [model-id] | load <model-id> | stop
connections list | add <harness> [--set-current <model-id>] [--install-skill]
connections sync [harness] | remove <harness>
docs [topic-id]
```

Removed output flags are not retained as aliases. The acquisition command remains `catalog pull`,
and there is no global JSON mode.

## Domain ownership

- `service` reports runtime readiness and login-startup intent.
- `hardware` reports the local inference topology, current memory use, and current allocation.
- `catalog` reports reviewed model choices, machine-specific assessment evidence, recommendations,
  and download operations.
- `models` reports models present or undergoing local operations and controls runtime residency.
- `connections` reports harness installation and durable Magnitude connection intent.

Catalog output never includes acquisition or residency state. Model-status output never includes
catalog ranking or provenance. The focused `models status <model-id>` view is the observation point
for download and load progress.

## Presentation

Comparable collections use borderless tables. Heterogeneous details use labeled fields. Mutations
use concise acknowledgements and include an exact observation command when work continues in the
background. A table that does not fit the terminal becomes labeled row blocks; canonical model and
harness IDs are never truncated.

Output uses friendly names as the primary identity and prints the exact canonical ID whenever an
object can be addressed by another command. Empty state is a successful, explicit sentence. Normal
errors write one actionable product message to stderr and exit nonzero. Redirected output contains
no cursor control, animation, or ANSI dependency.

Normal output excludes ACN/ICN terminology, assessment and environment IDs, package identities,
cache paths, native device indices, source revisions, raw tags, ranking utility, operation IDs,
retryability flags, and stack traces.

Memory uses hardware-conventional units; storage and transfer use decimal units; context uses
compact token counts; generation speed uses `tok/s`. Rounded values are presentation only.

## Catalog and recommendation behavior

`catalog list` displays only assessed catalog configurations that fit the current machine. It shows
friendly identity, predicted memory, baseline speed, configured context, speculative acceleration,
and canonical ID. Outstanding or failed assessments are summarized after the useful rows.

`catalog recommendations` reuses the shared onboarding eligibility and ranking policy. Preference
is one of Fastest, Faster, Balanced, Smarter, or Smartest and defaults to Balanced; limit defaults
to ten. Recommendation evidence includes speed, memory, context, intelligence, artifact accuracy,
acceleration, capabilities, and canonical ID. Raw ranking scores and aggregate utility remain
private. The command is a client projection over existing catalog and hardware authorities, not a
second server recommendation authority.

`catalog show` supplies one model's useful curated and machine-specific evidence.

No catalog or model observation command waits for assessment or residency to settle. Service health
and `service start` completion are likewise independent of background assessment.

## Model operations

`catalog pull` converges a catalog model to installed and current. It acknowledges admitted
download or update work and directs the caller to focused model status. Pull, cancel, and removal
validate only model-ID syntax before delegating directly to their authoritative mutations.

`models status` lists catalog-attributed and externally discovered models together when they are on
the computer or have relevant acquisition/removal work. One status field applies product priority:
removal, transfer, failures, load/stop, ready, update availability, then unloaded. The addressed
form reports installation, transfer progress, runtime, memory, context, and actionable failure
details without historical or internal operation state.

Load and stop delegate directly to their authoritative mutations. Magnitude has one active local
residency slot, so stop remains unaddressed.

## Connections

Connection observation distinguishes `Built in`, `Connected`, `Available`, and `Not installed`.
`Connected` comes from the durable connection manifest, not from executable detection. Connection
mutations delegate directly to the existing service operations. Handoff output is copyable shell
text rather than raw argv or environment structures, and Magnitude does not launch it from a
non-interactive command.

## Conformance

- Every public command and option has useful help.
- Collection ordering is deterministic and every collection has an explicit empty state.
- Addressable rows preserve exact canonical IDs at every terminal width.
- Recommendations match shared onboarding ranking and memory eligibility.
- No observation command waits for assessment or residency completion.
- Mutation commands acknowledge authoritative completion without adding preflight state
  interpretations.
- Agent documentation completes onboarding using only the human command surface.
- Tests cover collection, detail, narrow-width, empty, partial, failure, and redirected forms.
