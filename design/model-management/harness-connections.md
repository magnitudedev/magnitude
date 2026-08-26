---
applies_to:
  - packages/client-common/src/harness-connections/**
  - cli/src/harness-connections/**
  - cli/src/commands/connections.ts
  - cli/src/runtime/interactive.tsx
  - cli/src/features/model-setup/harness.tsx
---

# Harness connections

`HarnessConnection` is the internal product boundary for configuring external agent harnesses to
use Magnitude. “Connections” is only the public CLI noun. There is no internal connection-manager
entity.

## Registry and observation

One canonical priority list orders all consumers:

```text
Magnitude, Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, Cline
```

Detection proves that the harness executable is launchable. Detection searches the user's ambient
PATH after removing dependency-local `node_modules/.bin` entries, so running Magnitude from a
checkout cannot accidentally select a stale project dependency instead of the user's installed
harness. Magnitude is always selectable and is
presented first with “Optimized for local models” as its blue supporting label. Installed external harnesses follow in
priority order. Supported but absent harnesses stay in the same list, move below all selectable
rows, display Not installed, and are disabled.

## Durable connection and connector ownership

The user manifest records desired harness connections. It contains no gateway credential or
historical harness settings. Each entry contains the harness, complete installed Magnitude model
set as `{ id, name }` descriptors, optional current-model selection, and update time. `add` connects
every installed Magnitude model and optionally applies `setCurrent`; `sync` refreshes the complete model set without launching
anything; and `remove` deletes only values
that still equal the connector's expected managed value. Divergent user edits are preserved.
Harness defaults and global model selections are never changed, so removal has no prior value to
restore. Generic settings that are not uniquely Magnitude-owned remain untouched.

The manifest is durable user state with no global format version. Missing state defaults to no
connections. Recovery applies recursively at the nearest valid boundary: an invalid root resets to
an empty connection list after preserving the original, an invalid connection removes only that
entry, and an invalid entry property is removed or defaulted without discarding the entry whenever
its harness identity remains valid. Valid siblings survive every recovery step.

One `HarnessConnector` implementation exists for every supported harness. The connector owns its
identity, display metadata, executable, configuration targets, wire dialect, exact active-model
strategy, skill target, owned removal inverse, and launch plan. The registry is complete and follows
the canonical priority order. The harness-connection service owns only manifest intent,
installation observation, transactional file snapshots, registry dispatch, and shared skill/startup
orchestration. Onboarding and the public `magnitude connections` commands call this same service;
neither contains connector-specific transforms.

The unified connector contract is `detect`, `connect(HarnessConnectionSpec)`,
`disconnect(HarnessConnectionSpec)`, `launch(modelId, installation)`, and `installSkill`. The spec
contains the complete Magnitude model set as `{ id, name }` plus an optional `setCurrent` model ID. Each harness
implements this contract in its own module. The registry only composes those modules in canonical
priority order; it contains no harness configuration logic.

The protocol policy is deliberately narrow: every harness capable of consuming OpenAI Chat
Completions uses `http://127.0.0.1:10100/inference/v1/chat/completions`. Pi and Oh My Pi declare
`openai-completions`; OpenCode uses `@ai-sdk/openai-compatible`; OpenClaw declares
`openai-completions`; Hermes uses a named `custom:magnitude` provider with its
`chat_completions` transport; and Cline uses its
default OpenAI-compatible provider path rather than opting into Responses.

Codex is the sole Responses consumer. Its separately owned profile uses the base
`http://127.0.0.1:10100/inference/v1`, producing `/inference/v1/responses`, with
`wire_api = "responses"`. The profile does not impose a reasoning effort. Claude Code is the
sole Anthropic Messages consumer. It uses
`http://127.0.0.1:10100/inference/anthropic`, producing `/inference/anthropic/v1/messages`, and
receives `anthropic-local/<model-id>` through launch-scoped model environment and `--model`. The reserved
prefix selects the local Anthropic route instead of the byte-preserving upstream route. Onboarding
always supplies `setCurrent` for its selected model, so every external handoff starts on that exact
model. A CLI connection without `--set-current` changes no current-model selection. Provider-local
registrations expose the complete Magnitude model set.
The model name is Magnitude's catalog display name followed by its variant label in parentheses.
Pi, OpenCode, OpenClaw, Oh My Pi, and Cline receive that name through their native per-model display
field. Hermes, Codex, and Claude Code have no separate display-name field in the configuration path
used by their connector, so those harnesses receive the provider-facing model ID only.
For Cline, the connector writes its supported `openai-compatible` provider settings, its named model
catalog in `<CLINE_DATA_DIR>/settings/models.json`, and the launch
plan passes `--provider openai-compatible --model <id>` in addition to `--tui`; configuration
presence alone is not treated as active-model proof.
OpenClaw launches `openclaw tui --local --session agent:magnitude:main`, using its embedded runtime so handoff neither requires
nor starts an OpenClaw Gateway daemon. Its connector owns a dedicated `magnitude` agent whose model
is the selected Magnitude model and launches the `agent:magnitude:main` session. It does not change
the user's global primary model or other agents.

## Skill and startup options

Skill installation writes one independently managed `magnitude/SKILL.md` into only the selected
harness's native user-wide skill root. It never fans out to other detected harnesses and never
overwrites an unmanaged conflicting skill.

“Launch Magnitude on startup” means idempotently registering the per-user service for the user's
operating-system session. The visible copy remains “startup”; implementation and diagnostics may
describe the platform mechanism as a user login service. Startup registration and skill
installation complete before harness configuration and handoff.

## Handoff

An adapter returns an executable path, argv, environment additions, and exact model ID. Magnitude
does not invoke a shell. The interactive runtime closes its client, releases renderer ownership,
restores terminal state through scoped finalizers, and then runs the child with inherited standard
I/O and working directory. A child TUI and the Magnitude renderer never own the terminal together.
