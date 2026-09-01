---
applies_to:
  - packages/client-common/src/harness-connections/**
  - cli/src/harness-connections/**
  - cli/src/commands/connections.ts
  - cli/src/commands/connections-runtime.ts
  - cli/src/server/service.ts
  - cli/src/runtime/interactive.tsx
  - cli/src/features/model-setup/harness.tsx
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

The shared service owns manifest persistence, installation observation, transactional file
snapshots, connector dispatch, skill installation, and startup orchestration. It contains no
harness-specific configuration transforms.

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
manifest selection state. A launch plan contains an executable, arguments, environment additions,
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

## Conformance

A conforming connector must prove that:

- its generated configuration is accepted by the supported harness version;
- every published model remains independently selectable after connection and sync;
- ordinary independent launch uses the persisted Magnitude model when one was explicitly selected;
- sync replaces stale Magnitude model metadata without changing unrelated user state;
- disconnect removes only Magnitude-owned state and conditionally restores selection; and
- reasoning behavior matches the projection table across startup, persisted state, session override,
  model switching, and direct TUI launch.
