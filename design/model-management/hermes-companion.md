---
applies_to:
  - integrations/hermes/**
  - cli/src/harness-connections/connectors/hermes*.ts
  - packages/release/src/hermes-plugin*.ts
---

# Hermes companion

The companion supports the ordinary Hermes installation and launch command. It must not replace
the host, patch private methods, select another terminal renderer, or require a fork. Installing
the companion does not select a model, install a private CLI, contact inference, or start a service.

Hermes owns conversations, provider selection, credentials, inference parsing, and terminal layout.
Magnitude owns model management and inference progress. The companion crosses Magnitude's public
ACN boundary; it never calls private ICN management routes or parses CLI model-command output.

## Model controls

The native Python adapter implements finite model RPCs from a generated subset of the canonical
ACN contract. Operation names, payload/result schemas, protocol version, and recovery policy come
from that contract, not a second handwritten model protocol. Schema validation is performed at the
adapter boundary; ACN remains authoritative for domain validation.

Connection is lazy. An explicit model command may run `magnitude service start` when the service
is absent. A protocol mismatch is reported with repair instructions and never replays a mutation.
Every request is fenced to the observed ACN instance. A lost mutation response has an unknown
outcome; neither reconnect nor presentation retry may submit it again automatically.

The adapter has bounded request, framing, and subprocess lifetimes. Errors are actionable text,
not tracebacks or successful empty results. Loading an unavailable service cannot prevent the
Hermes plugin from loading or its setup command from working.

## Installation ownership

Magnitude's connection flow installs the release-selected Git revision through the native Hermes
package manager with security scanning enabled. It does not use `--force`, a host patch, or a
private CLI dependency. The same package supplies the Python companion, Desktop component, and
fallback skill. Desktop contribution enablement remains an explicit host UI choice.

A connection receipt distinguishes a Magnitude-owned native package from a borrowed pre-existing
package. A borrowed package must have verified content and a matching RPC version; it is never
replaced or removed. For an owned package, update/removal verifies content, native source/revision
metadata, and the complete file set. Extra user files, symlinks, changed provenance, or modified
consumer bytes prevent deletion. Runtime Python bytecode and Git metadata are not user content.

Native mutations register compensation before running. An upgrade removes only the verified old
installation and installs the new pin, with restoration to the old exact pin if later work fails.
Enablement changes own only Magnitude's membership in the enabled/disabled lists and preserve
unrelated entries. The independently installed skill is outside package ownership.

## Instructions and setup

The package bundles the canonical Magnitude skill. An independently installed flat Magnitude
skill takes precedence; package installation must not overwrite it or introduce a second copy
into the user's skill directory. Package-provided instructions use Hermes's namespaced skill API.

Explicit setup presents the canonical onboarding prompt for the user to submit. There is no
automatic conversation injection. Startup must not ask repeatedly, interrupt active work, or
consume a user's editor contents.

## Presentation boundary

The ordinary terminal interface does not currently expose a supported plugin working-row or
below-input widget API. Model controls and instructions must work there without enabling an
alternate renderer. A terminal plugin must not simulate a persistent row by printing ANSI cursor
controls into a terminal owned by Hermes.

Ordinary terminal status uses prompt_toolkit's public above-application printing API, captured
from an active terminal callback. Phase transitions and a final summary appear in scrollback;
per-token/per-percentage updates do not flood it. The host retains and redraws the editor. No
terminal application means no terminal output: gateway, Desktop, and headless streams remain clean.
Completion or cancellation closes the request's output lifetime before late observations can print.

Desktop presentation may use its public status-bar contribution and focused-session ownership
API. It must not attribute another session's inference to the focused conversation. Missing
observability is unavailable progress, not fabricated work or successful completion.

## Conformance

- Plugin loading and setup work with no CLI, service, or local model installed.
- Model controls validate the generated wire contract and preserve at-most-once semantics.
- A native install followed by plain `hermes` makes the commands available without changing the
  renderer, selected model, or provider.
- Shared skills survive companion installation and removal.
- Inference and ordinary host behavior remain functional when presentation fails.
