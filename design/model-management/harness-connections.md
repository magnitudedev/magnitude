---
applies_to:
  - packages/client-common/src/harness-connections/**
  - cli/src/harness-connections/**
  - cli/src/commands/connections.ts
  - cli/src/commands/connections-runtime.ts
  - cli/src/server/service.ts
  - cli/src/runtime/interactive.tsx
  - cli/src/features/model-setup/harness.tsx
  - integrations/pi/**
  - scripts/dev-pi.ts
  - package.json
---

# Harness connections

`HarnessConnection` configures an external agent harness to use Magnitude. `Connections` is the
public CLI noun; there is no separate connection-manager domain.

## Observation

Harnesses have one canonical order:

```text
Magnitude, Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, Cline
```

Detection proves that an external executable is launchable from the user's ambient `PATH`, excluding
dependency-local binaries. Magnitude is always `Built in`. A manifest-owned external connection is
`Connected`, an installed unconnected harness is `Available`, and an absent harness is `Not
installed`. Installation alone does not imply connection.

## Ownership

The user manifest records connection intent, the complete installed Magnitude model descriptors,
optional model-restoration state, and update time. It contains no gateway credential. Descriptors
come from the enriched OpenAI model list and include model identity, presentation, limits,
capabilities, and the exact normalized reasoning domain and default. There is no second
Magnitude-specific model inventory.

Restoration has exactly one uniform field across connectors: `model`, the prior persistent model
selection. Connector-owned provider, endpoint, and catalog projections are removed rather than
stored as restoration state.

Each harness has one connector. A connector owns that harness's:

- Magnitude-specific provider, profile, agent, or catalog projection;
- native active-model representation and restoration;
- reasoning-control projection;
- inference protocol and endpoint;
- skill installation target; and
- launch plan.

The shared service owns manifest persistence, installation observation, transactional compensation,
connector dispatch, skill installation, and startup orchestration. It contains no
harness-specific configuration transforms.

A connector may require one harness-native companion package. That package is part of the
connection's desired state rather than an optional setup extra. The connector owns its exact
package identity, source, inspection, installation, activation, and removal operations. The shared
service owns package reconciliation, mutation ordering, compensation, and persistence of package
ownership. A connection records whether Magnitude installed the package or found it pre-existing,
plus a connector-specific validated receipt of any enablement fields Magnitude changed. Disconnect
removes a package only while Magnitude owns its installation; otherwise it conditionally restores
only those fields, never recreating a removed user entry or overwriting subsequent edits. Sync reconciles a
required package as well as connector configuration.

The Pi companion has one desired package source. Production uses its exact versioned npm source;
the repository development launcher supplies the local integration directory instead. The manifest
records the source actually installed. Reconciliation addresses that exact source, replaces a
Magnitude-owned package when the desired source changes, and never replaces a pre-existing
user-owned package merely because development mode is active.

Configuration presence is not proof of package availability. Pi reconciliation verifies the supported
host, installed package version, and extension entrypoint before reporting success. Filters follow
the supported host's native rules, including empty arrays, basename and absolute exclusions, and
autoload-disabled ordering. Relative local sources resolve from Pi's settings directory; explicit
agent-directory overrides govern both configuration and native package commands. An incompatible
borrowed package is reported, not silently replaced.

Connection mutations hold one cross-process lock through fresh manifest observation, native
operations, configuration, and manifest commit. Process death releases the lock; elapsed time never
transfers ownership. Compensations are registered before mutations, run in reverse order, and all
are attempted even after a recovery failure. Failed operations report incomplete recovery together
with the original error. File recovery checks the value written by the transaction before restoring
it, preserving concurrent edits. Manifest commit is the uninterruptible durability point.

Applying a connection is an idempotent replacement of Magnitude-owned state. Unrelated user state
is preserved. Without an explicit model, connection and sync do not change the harness's current
model or generic reasoning preferences. With an explicit model, the connector records the previous
selection once and persists the Magnitude model used by an ordinary new harness session.

`add` publishes the complete installed model set and may persist a selected model. `sync` refreshes
the model set without launching or changing selection. `remove` deletes Magnitude-owned state and
conditionally restores the prior model only while the current selection remains Magnitude-owned.
Cleanup continues when restoration is unnecessary or unsafe.

Manifest recovery preserves the nearest valid boundary: invalid properties are removed or
defaulted, an invalid connection removes only that connection, and an invalid root is preserved for
diagnosis before resetting to no connections. Valid siblings survive.

## Configuration and launch

Configuration and launch are independent. Configuration publishes the complete model projection
and optional persisted selection. Launch receives an exact model ID directly and must not depend on
manifest selection state. A launch plan contains the exact detected executable for programmatic
launch, the stable ambient command name for user-facing handoff, arguments, environment additions,
and model identity; it never invokes a shell.

The interactive runtime releases its inference client and terminal renderer before running the
child with inherited standard I/O and working directory. Magnitude and the child TUI never own the
terminal concurrently.

Connectors preserve canonical model IDs as opaque keys. Claude-facing identities add only the
reserved `anthropic-local/` routing prefix. Provider-local registrations expose every installed
Magnitude model, including callable external Hugging Face models.

The protocol assignments are:

| Protocol | Harnesses |
| --- | --- |
| OpenAI Chat Completions | Pi, Oh My Pi, OpenCode, OpenClaw, Hermes, Cline |
| OpenAI Responses | Codex |
| Anthropic Messages | Claude Code |

## Reasoning-effort correctness

Reasoning validity is model-relative. Magnitude owns each model's exact normalized effort domain
and default. A connector owns the projection between that domain and its harness's native controls;
there is no universal harness control vocabulary.

Harness projection and inference safety are independent obligations:

1. **Precise projection:** every connector must express the model's exact domain and default as
   faithfully as its harness permits. Global admission safety is not permission to publish generic,
   stale, or less precise controls.
2. **Global admission safety:** every locally admitted request applies `RoundUpOrClamp` after model
   resolution. Requests do not carry a trusted harness identity, so correctness cannot depend on
   selecting a harness-specific admission mode.

Projection completeness is normative:

| Harness | Projection | Reason |
| --- | --- | --- |
| Magnitude | Complete | A slot persists model and effort together and normalizes them against the catalog. |
| Pi | Complete | Per-model thinking maps govern initial selection, session restoration, and model changes. |
| Oh My Pi | Complete | Native per-model thinking profiles govern selection and model changes. |
| OpenCode | Complete | Reasoning variants are model-relative; an unavailable variant is not serialized. |
| OpenClaw | Complete | Per-model thinking maps resolve agent, session, and model-change state before serialization. |
| Hermes | Best available | A model-agnostic session reasoning value can override per-model configuration. |
| Cline | Best available | Its TUI can select fixed generic efforts outside the model's advertised domain. |
| Codex | Best available | Startup configuration and raw overrides can bypass model-relative catalog validation. |
| Claude Code | Best available | Sticky effort state is generic and gateway discovery cannot publish a per-model effort domain. |

Complete projection means the harness already emits a supported model-specific effort; it is not
another name for rounding. Best-available connectors still publish every exact capability their
harness can represent and persist the most precise safe default available.

At admission, an exact supported effort remains unchanged and omission selects the model default.
`none` is valid only when the model supports disabling reasoning. The
[protocol compatibility design](../inference/http-protocol-compatibility.md) owns the global
`RoundUpOrClamp` admission invariant. The canonical inference request and provider model contract do
not own this wire-compatibility behavior.

## Harness-specific requirements

- **Codex:** install one HTTP and native Responses WebSocket proxy provider named `OpenAI`, using
  Codex's native OpenAI auth, for both bundled and local entries. Its base URL is
  `/inference/v1/proxies/codex`; the gateway routes each request frame by its selected model and
  never translates WebSocket events into SSE. Publish one catalog composed from the installed Codex binary's opaque
  bundled entries plus `magnitude-local/` entries. An explicit selection persists that local alias
  as Codex's ordinary startup model. Sync re-exports the installed binary's catalog without
  changing selection or restoration state. Disconnect removes the proxy provider, preserves a
  newly selected bundled OpenAI model on the built-in provider, clears the owned catalog reference,
  and restores the prior model selection only while the current selection remains Magnitude-owned.
- **Hermes:** publish per-model defaults without changing unrelated global preferences. Session
  precedence still requires the Chat Completions boundary.
- **Cline:** publish its supported OpenAI-compatible provider, exact model metadata, and persistent
  provider/model selection. Its fixed TUI effort surface still requires the Chat Completions
  boundary.
- **Claude Code:** persist the Magnitude gateway settings and selected `anthropic-local/` model, but
  no effort default. Its discovery schema limitation requires the Anthropic boundary.
- **OpenClaw:** use a dedicated Magnitude agent and a fresh agent-scoped session for handoff so stale
  session model overrides cannot replace the selected model. Explicit connection selection also
  updates the ordinary global primary with conditional restoration.

## Skills and startup

Magnitude, Pi, OpenCode, OpenClaw, Codex, and Oh My Pi share the `~/.agents/skills` target. Hermes,
Claude Code, and Cline use harness-specific user targets. Installation atomically replaces the
selected target's Magnitude skill with the bundled version; shared targets receive one physical
copy.

Startup means idempotently registering Magnitude's per-user operating-system service. Startup and
skill installation finish before connector configuration and handoff. Codex and Claude Code's
persistent proxy configuration require this service; disconnect removes those settings only while
they retain Magnitude's installed values.

The Magnitude skill is independent of a harness companion package. It remains optional for ordinary
connectors, but is required for Pi because catalog discovery, recommendation, acquisition, and
removal are agent-guided rather than duplicated as Pi extension commands. Connecting Pi always
installs or enables the desired Magnitude Pi package through Pi's package command and installs the
skill into Pi's shared agent-skill target. Both the
non-interactive `connections add pi` flow and interactive onboarding submit the same connection
request to the shared service; neither presentation surface owns a second installation path.
Interactive onboarding discloses the exact package source and that Pi extensions execute with the
user's authority. A successful connection reports whether an already-running harness must reload
or restart.

During a Magnitude request, the Pi companion uses Pi's native working row rather than an extension
footer status. Model loading and prefill temporarily replace the generic working message; generation
is presented as timed work. The companion treats transport requests and a Pi agent run as separate
lifecycles: request progress owns the live row, while `agent_start` through `agent_settled` owns the
retained summary. Completion restores Pi's default working message and presents the model display
name, total agent-run wall time, the first request's time to first token, and token-weighted generation
throughput in one widget line immediately above the editor. Pi's stock parser decides semantic
success; HTTP EOF alone cannot authorize a summary. Responses and their retry attempts are tracked
independently, including overlapping and delayed observations. Timings are cumulative snapshots;
throughput sums tokens and decode time once for each successful response's final request. Run duration
uses monotonic time. Starting another run, cancellation, failure, switching providers, or extension disposal clears
the retained summary and restores Pi's default working message.

The extension owns a scoped observer, live-row timer, and subprocess lifetime. Disposing it cancels
pending work; terminal request handles and older runs cannot mutate newer presentation. Presentation
failures do not prevent inference. Status completion lookups share in-flight work and briefly cache
ready discovery, but never cache initialization or failure. Explicit model selection refreshes status.
Loading is acknowledged only after server readiness, with a longer bound than discovery or stop.

The repository exposes one `dev:pi` entrypoint. It selects an installed Magnitude model, connects
Pi through the ordinary connection service using the local package source, and launches
Pi with a scoped executable for the current source CLI. Temporary executables live outside the
repository and remain available for the entire child session. This development path exercises the
same provider configuration, package ownership, skill installation, and Pi extension loading as a
published connection. Pi's user configuration, connection receipts, and bundled skill are isolated
in the development scope. Pi inherits the caller's working directory: project files and context
remain available, and development setup never substitutes a temporary workspace.
Automatic skill discovery is disabled for this launcher; only the explicit checkout skill is loaded,
including after reload. Changing Pi's agent directory alone does not isolate shared agent skills.
It builds the extension and runs the checkout's inference runtime and suppresses successful
native-build diagnostics while preserving complete failure diagnostics. It inherits the caller's
environment but does not start a telemetry collector or enable tracing itself.
It borrows the fixed service endpoint only from a stopped state or the installed service manager:
after Pi exits, it stops the development daemon and restores whether the installed service was
running. An already-running unmanaged daemon is rejected because its launch state cannot be safely
reconstructed.

## Conformance

A conforming connector must prove that:

- its generated configuration is accepted by the supported harness version;
- every published model remains independently selectable after connection and sync;
- ordinary independent launch uses the persisted Magnitude model when one was explicitly selected;
- sync replaces stale Magnitude model metadata without changing unrelated user state;
- disconnect removes only Magnitude-owned state and conditionally restores selection; and
- required companion packages are reconciled transactionally, user-owned packages survive
  disconnect, and every connection entry accurately records package ownership; and
- Pi connection, sync, source replacement, and removal address the exact recorded local or npm
  package source, and the local development entrypoint leaves no repository artifacts; and
- reasoning behavior matches the projection table across startup, persisted state, session override,
  model switching, and direct TUI launch.
