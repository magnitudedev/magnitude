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
catalog status | list | show <model-id> | recommendations [--preference <value>] [--limit <count>]
catalog pull <model-id> | cancel <model-id> | remove <model-id>
models status [model-id] | load <model-id> | stop
connections list | add <harness> [--set-model <model-id>] [--install-skill]
connections sync [harness] | remove <harness>
docs [topic-id]
```

Removed output flags are not retained as aliases. The acquisition command remains `catalog pull`,
and there is no global JSON mode. The plugin-facing `models status`, `models load`, and `models stop`
commands independently accept `--json`.

## Domain ownership

- `service` reports runtime readiness and login-startup intent.
- `hardware` reports the local inference topology, current memory use, and current allocation.
- `catalog` reports model discovery and assessment progress, reviewed model choices,
  machine-specific assessment evidence, recommendations, and download operations.
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

## Plugin-facing JSON output

`--json` is a versioned machine contract for model observation and residency control. It is scoped
to `models status`, `models load`, and `models stop`; it is not inherited globally and does not
change the default human presentation.

Every JSON invocation writes exactly one compact JSON document followed by one newline. Success
writes only to stdout and exits successfully. Failure writes only to stderr and exits nonzero.
Neither stream contains prose, ANSI escapes, progress animation, or additional JSON documents.

Every document has these common fields:

- `schemaVersion`, currently the integer `1`;
- `command`, exactly `models.status`, `models.load`, or `models.stop`;
- `ok`, the success discriminator; and
- either `data` when `ok` is true or `error.message` when `ok` is false.

The envelopes are exactly:

```text
{ schemaVersion: 1, command, ok: true, data }
{ schemaVersion: 1, command, ok: false, error: { message } }
```

The version covers the complete JSON document, including command data. Any structural or semantic
contract change—including adding, renaming, or removing a field—requires a new version. An
implementation correction that leaves every documented meaning and shape intact does not.

`models status --json` preserves the human command's observation semantics and deterministic model
ordering. Its data distinguishes initialization from a ready list or ready addressed detail. A model
contains its canonical ID, friendly display name, source, optional byte-accurate memory requirement,
optional exact context length, installation state, and optional residency state. Installation and
residency are separate discriminated unions. Transfer states contain their native stage, completed
and total bytes, and optional bytes per second. Loading contains its native stage and optional
fractional progress. Failure states contain a product message and, for residency failures, whether
retry is meaningful. Optional fields are absent rather than `null`.

Status data has exactly these top-level variants:

```text
{ state: "initializing", view: "list" | "detail" }
{ state: "ready", view: "list", models: Model[] }
{ state: "ready", view: "detail", model: Model }
```

Each `Model` has `modelId`, `displayName`, `source` (`catalog` or `discovered`), `installation`, and
optional `memoryBytes`, `contextLength`, and `residency`. The exact installation discriminators are
`not_installed`, `installing`, `install_failed`, `installed`, `update_available`, `updating`,
`update_failed`, `removing`, `remove_failed`, and `unavailable`. The exact residency discriminators
are `unloaded`, `requested`, `loading`, `ready`, `stopping`, and `failed`.

`installing` and `updating` carry `progress: { stage, completedBytes, totalBytes, bytesPerSecond? }`.
Installation failures carry `error: { message }`. `loading` carries `{ stage, progress? }`, where
progress is a number from zero through one. `stopping` carries its release `reason`. Residency
failure carries `error: { message, retryable }`. No other variant carries those state-specific
fields.

The unaddressed status form includes the same models as human output: discovered models and catalog
models that are installed or have relevant acquisition/removal work. The addressed form may return
a known not-installed catalog model because it observes the exact requested identity. Initialization
does not wait for readiness. Unknown or malformed IDs are failures once the authoritative catalog is
available, preserving existing command behavior.

`models load --json` succeeds after the authoritative load mutation is admitted and returns the
canonical `modelId` with outcome `load_requested`; it does not claim that loading has completed.
`models stop --json` succeeds after the authoritative stop mutation and returns outcome `stopped`.
Callers observe continuing work through `models status --json` rather than polling an internal
operation identity.

## Catalog and recommendation behavior

`catalog status` reports the authoritative discovery and assessment completion flags and their
progress counts independently. It does not infer completion from catalog rows or recommendation
availability and does not wait for either phase to finish.

`catalog list` displays only assessed catalog configurations that fit the current machine. It shows
friendly identity, predicted memory, baseline speed, configured context, speculative acceleration,
and canonical ID. Outstanding or failed assessments are summarized after the useful rows.

`catalog recommendations` reuses the shared onboarding eligibility and ranking policy. Preference
is one of Fastest, Faster, Balanced, Smarter, or Smartest and defaults to Balanced; limit defaults
to ten. Recommendation evidence includes speed, memory, context, intelligence, artifact accuracy,
acceleration, capabilities, and canonical ID. Raw ranking scores and aggregate utility remain
private. The command is a client projection over existing catalog and hardware authorities, not a
second server recommendation authority. Successful nonempty output ends with the exact local
documentation command for interpreting the evidence and ranking methodology.

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
text rather than raw argv or environment structures. It uses the harness's ordinary ambient command
name rather than the exact detected executable path, and Magnitude does not launch it from a
non-interactive command.

## Conformance

- Every public command and option has useful help.
- Collection ordering is deterministic and every collection has an explicit empty state.
- Addressable rows preserve exact canonical IDs at every terminal width.
- Catalog status preserves the authoritative discovery and assessment completion states.
- Recommendations match shared onboarding ranking and memory eligibility.
- No observation command waits for assessment or residency completion.
- Mutation commands acknowledge authoritative completion without adding preflight state
  interpretations.
- Agent documentation completes onboarding using only the human command surface.
- Tests cover collection, detail, narrow-width, empty, partial, failure, and redirected forms.
- Plugin-facing JSON tests cover every installation and residency variant, list and addressed views,
  initialization, empty state, deterministic ordering, mutation acknowledgements, structured
  failures, stream separation, newline framing, and schema validation.
- Enabling JSON cannot change command effects, validation order, mutation admission, or exit status.
