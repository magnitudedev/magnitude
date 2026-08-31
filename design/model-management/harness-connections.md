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

Non-interactive connection observation combines executable detection with durable manifest intent.
It reports Magnitude as `Built in`, a manifest-owned external connection as `Connected`, an
installed unconnected harness as `Available`, and an absent harness as `Not installed`. Installation
alone never implies a connection.

## Durable connection and connector ownership

The user manifest records desired harness connections. It contains no gateway credential or
historical harness settings. Each entry contains the harness, complete installed Magnitude model
set as harness-facing descriptors, optional handoff-model selection, and update time. The CLI derives
each descriptor through the SDK from the enriched OpenAI model-list `data` entry; no second
Magnitude-only model-list array exists. A descriptor contains identity, display name, description,
context window, advertised output ceiling, modalities and tool capabilities, and the model's
reasoning domain and default. Existing persisted descriptors without an output ceiling migrate to
the lesser of their context window and 32,768 tokens. `add` connects every installed Magnitude model and may
select one for the immediate handoff; `sync` refreshes the complete model set without launching
anything; and `remove` deletes the connector's Magnitude-owned provider, agent, profile, or catalog.
Applying a connection is an unconditional idempotent upsert of Magnitude-owned state. Existing,
older, partial, or differently shaped Magnitude state is replaced rather than classified as a
conflict. Harness defaults and global model selections are never changed. Generic settings that are
not uniquely Magnitude-owned remain untouched, except for Claude Code's two documented gateway
settings.

The manifest is durable user state with no global format version. Missing state defaults to no
connections. Recovery applies recursively at the nearest valid boundary: an invalid root resets to
an empty connection list after preserving the original, an invalid connection removes only that
entry, and an invalid entry property is removed or defaulted without discarding the entry whenever
its harness identity remains valid. Valid siblings survive every recovery step.

One `HarnessConnector` implementation exists for every supported harness. The connector owns its
identity, display metadata, executable, configuration targets, wire dialect, exact active-model
strategy, skill target, owned removal inverse, and launch plan. The registry is complete and follows
the canonical priority order. The harness-connection service owns only manifest intent,
installation observation, transactional file snapshots, registry dispatch, central skill
installation, and startup orchestration. Onboarding and the public `magnitude connections`
commands call this same service; neither contains connector-specific transforms.

The unified connector contract is `detect`, `connect(HarnessConnectionSpec)`,
`disconnect(HarnessConnectionSpec)`, and `launch(modelId, installation)`, plus one declarative skill
installation target. The spec contains the complete harness-facing Magnitude model descriptors plus
an optional `setCurrent` model ID. During sync it also contains the prior manifest descriptors so a
connector can replace only its previous model-relative projection. Each harness implements this
contract in its own module. The
registry only composes those modules in canonical priority order; it contains no harness
configuration logic.

Reasoning validity is model-relative. Magnitude owns the exact normalized model domain; every
connector separately owns its harness's control vocabulary and the projection from those controls
to normalized efforts. Shared projection logic may operate on a connector-supplied control surface,
but no common list is treated as universally native. A projected default is always a harness
control whose mapping resolves to the model default, never an effort name copied into a control
field. Connect and sync preserve user-owned global and session preferences. Launch selects only the
provider/profile/agent and exact model, so the same persisted projection applies when a connected
harness is opened independently. A fixed-scale harness may give a differently named control to a
normalized named mode, or map its sole enabled control to a model's sole enabled behavior. ICN never
rounds an ordinary explicit unsupported OpenAI effort. If a user bypasses native validation, ICN
returns the supported-domain error.

Pi and OpenClaw receive per-model thinking-level maps, Cline receives per-model reasoning options,
OpenCode receives exact model variants, Oh My Pi receives native thinking configurations, Codex
receives exact catalog domains/defaults, and Hermes receives per-model default overrides that take
precedence over its preserved global effort. Magnitude already stores model and effort together per
slot. A connector does not publish `none` when the model cannot disable reasoning.

The protocol policy is deliberately narrow: every harness capable of consuming OpenAI Chat
Completions uses `http://127.0.0.1:10100/inference/v1/chat/completions`. Pi and Oh My Pi declare
`openai-completions`; OpenCode uses `@ai-sdk/openai-compatible`; OpenClaw declares
`openai-completions`; Hermes uses a named `custom:magnitude` provider with its
`chat_completions` transport; and Cline uses its OpenAI-compatible client and Chat Completions
protocol inside a Magnitude-owned isolated data directory.

Codex is the sole Responses consumer. Its separately owned profile uses the base
`http://127.0.0.1:10100/inference/v1`, producing `/inference/v1/responses`, with
`wire_api = "responses"`. The connector also owns a Codex-native model catalog derived from the
shared descriptors and points the profile's `model_catalog_json` at it. This prevents fallback
metadata and supplies Codex with exact display names, context windows, input modalities, and
reasoning domains. When `setCurrent` is present, the profile selects that model; Codex obtains its
advertised default from the catalog instead of a profile-level `model_reasoning_effort` that could
become stale after model switching. The profile explicitly selects the default service tier so unrelated
global priority-tier configuration cannot leak into local requests. Claude Code is the
sole Anthropic Messages consumer. It uses
`http://127.0.0.1:10100/inference/anthropic`, producing `/inference/anthropic/v1/messages`, and
receives `anthropic-local/<model-id>` through `--model`. The reserved prefix selects the local
Anthropic route instead of the byte-preserving upstream route. Onboarding always supplies
`setCurrent` for its selected model, so every external handoff starts on that exact model through
the connector's launch plan. The optional `--set-current <model-id>` CLI input supplies
`setCurrent`, selects that model, and returns the connector's launch plan; omitting it leaves the
current-model selection unchanged.
Provider-local registrations expose the complete Magnitude model set.
That set includes callable external Hugging Face models using their exact canonical `hf:` identity.
Connectors preserve the identity as an opaque model key; Claude-facing discovery adds only the
reserved `anthropic-local/` routing prefix and removes only that prefix before native inference.
The model name is Magnitude's catalog display name followed by its variant label in parentheses.
Pi, OpenCode, OpenClaw, Oh My Pi, and Cline receive that name through their native per-model display
field. Codex receives it through its connector-owned native model catalog. Hermes and Claude Code
have no separate display-name field in the configuration path used by their connector, so those
harnesses receive the provider-facing model ID only.
For Cline, arbitrary provider IDs can appear in its local catalog but are not executable by its
agent runtime. The connector therefore writes Cline's supported `openai-compatible` provider and
named model catalog inside the dedicated `~/.magnitude/harnesses/cline` data directory. The launch
plan passes that directory through `--data-dir` together with `--provider openai-compatible --model
<id>` and `--tui`. The user's normal Cline data directory and provider settings remain untouched.
OpenClaw launches `openclaw tui --local --session agent:magnitude:main`, using its embedded runtime so handoff neither requires
nor starts an OpenClaw Gateway daemon. Its connector owns a dedicated `magnitude` agent whose model
is the selected Magnitude model and whose native `thinkingDefault` is the OpenClaw control that maps
to that model's advertised default. Per-model thinking maps clamp that agent preference after later model switches. The
connector launches the `agent:magnitude:main` session and does not change the user's global primary
model or other agents.

## Skill and startup options

Skill installation addresses a physical installation target rather than treating every harness as
the owner of a distinct copy. Magnitude, Pi, OpenCode, OpenClaw, Codex, and Oh My Pi select the
shared `~/.agents/skills` target. Hermes, Claude Code, and Cline select their harness-specific
user-wide targets because they do not discover the shared target by default. Selecting multiple
harnesses that share a target therefore retains one physical `magnitude/SKILL.md`; installation
never fans out to other targets.

The central installer owns target resolution and atomic publication. Every explicit installation
replaces the selected target's `magnitude/SKILL.md` with the current bundled skill, including when a
file is already present. Skill installation is therefore refresh, not create-if-absent behavior.
Interactive setup and `connections add --install-skill` invoke that same installer before applying
the harness configuration. The installed skill routes agents to the CLI-bundled onboarding topic so
all harnesses receive one portable CLI-only workflow rather than harness-specific instructions.

“Launch Magnitude on startup” means idempotently registering the per-user service for the user's
operating-system session. The visible copy remains “startup”; implementation and diagnostics may
describe the platform mechanism as a user login service. Startup registration and skill
installation complete before harness configuration and handoff.

Claude Code is the sole connector that must persist generic harness settings. Connecting it merges
`ANTHROPIC_BASE_URL=http://127.0.0.1:10100/inference/anthropic` and
`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` into the user-wide `settings.json` `env` object and
ensures Magnitude's per-user service is registered for login. It writes no credential, model,
model-alias, or attribution default. An ordinary later `claude` launch therefore discovers local
`anthropic-local/` aliases through Magnitude while retaining the user's normal Claude authentication
and model selection. Disconnect removes either setting only while it still equals Magnitude's
installed value, preserving subsequent user edits.

Claude Code 2.1.175 gateway discovery retains only model ID, display name, and description; it has
no discovery field through which Magnitude can install a per-model effort domain. Magnitude therefore
does not write or launch with a Claude effort. Both a Magnitude handoff and a later manual Claude
selection pass Claude's sticky `output_config.effort` through the local Anthropic adapter. Exact
supported values remain exact. When the selected model exposes exactly one enabled behavior, a
different enabled Claude label maps to that behavior. For a multi-effort model, an unsupported
ordinal label rounds upward to the least supported enabled ordinal or clamps to the greatest one
when the requested label exceeds the model's range. This `RoundUpOrClamp` rule is a scoped
Anthropic/Claude protocol translation, not generic gateway request rewriting. Disabled thinking
still requires the model to support `none`.

## Handoff

An adapter returns an executable path, argv, environment additions, and exact model ID. Magnitude
does not invoke a shell. The interactive runtime closes its client, releases renderer ownership,
restores terminal state through scoped finalizers, and then runs the child with inherited standard
I/O and working directory. A child TUI and the Magnitude renderer never own the terminal together.
