---
applies_to:
  - web/src/app.tsx
  - web/src/commands/**
  - web/src/components/**
  - web/src/hooks/use-menu-actions.ts
  - web/src/state/web-atoms.ts
  - web/src/stores/**
  - web/src/styles/**
  - desktop/src/renderer.tsx
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - packages/client-common/src/local-models/**
  - packages/client-common/src/model-slots/**
---

# Web local inference and appearance

The retained browser client consumes local model services. The inference-focused desktop has its
own shell, specified in `desktop-inference.md`, and shares the established appearance system and UI
primitives. Onboarding belongs exclusively to that desktop shell.

## Authority and boundaries

The unified `ModelCatalog` is ACN's read-only Magnitude product projection: each local row's
`acquisitionState` carries the model's complete materialization lifecycle (disk truth, transfer
progress, unacknowledged failure, update availability, and residency once installed) alongside
assessment presentation, provider availability, ranking scores, and product warnings. Native ICN
Models, Packages, Downloads, Instances, and Hardware remain authoritative beneath ACN and are not
client-visible. `ModelSlotsState` owns durable selection, favorites, and recency, resolved
server-side to truthful Slot states including residency.

Web consumes those domains through one connection-scoped Effect Query runtime containing ACN RPC
operations, plus the composed `LocalModels` and `ModelSlots`
client-common services. React hooks are adapters to those services. DOM components may derive
labels and layout, but do not construct services, cache server snapshots, or infer compatibility,
availability, readiness, progress, or command completion.

The web product presents only local models. Protocol support for another provider does not make it
a web product choice or readiness signal. Cloud login, account usage, subscription, connection,
and cloud-model surfaces are absent.

## Product flow

Browser startup presents the SDK daemon lifecycle before the ordinary application shell. The
browser has no onboarding gate or `/setup` command. Desktop onboarding and process ownership are
specified by their dedicated contracts. Neither surface fabricates startup progress or readiness.

The ordinary shell contains a dedicated Settings surface for local inference:

- Models is the searchable installed-model library. It lists downloaded model artifacts and exposes
  only artifact-level actions: reveal the daemon-published installed target path or remove the
  download. Externally managed Hugging Face cache artifacts may be revealed but do not expose a
  removal action. Slot selection, residency, favorites, transfer activity, and load controls do not belong
  on this surface.
- Catalog presents the unified assessed local catalog. Its index may be searched by model identity,
  filtered to installed models, and sorted by intelligence (the default), release date, download
  size, or name. Onboarding preference is not applied to this general catalog. Ordinary
  downloadable rows do not repeat an `Available` label; non-default lifecycle and compatibility
  states remain visible while completed `DoesNotFit` and `Incompatible` assessments are excluded
  from the browsable catalog. Catalog owns
  install, update, and transfer cancellation; active-model selection remains in the composer and
  installed-artifact removal remains in Models.
- Hardware presents server-reported topology and a labeled physical-memory breakdown alongside
  resident allocations. Internal admission thresholds are not exposed as end-user concepts.

Models and Catalog distinguish an unobserved query, server-side inventory initialization,
catalog discovery, a successfully loaded empty collection, and failure. Before the first snapshot,
and while the corresponding server lifecycle is still loading, each surface renders an explicit
loading state rather than empty model chrome. An empty-state message is shown only after the daemon
has made that collection authoritative. Partial usable choices may remain available while a refresh
or discovery operation continues, accompanied by its nonterminal state.

Opening Settings changes the application sidebar from session navigation to Settings navigation.
Models, Catalog, and Hardware are vertical sidebar destinations; the main pane renders only the
selected destination. Returning from Settings restores the session sidebar. On narrow layouts the
same navigation is presented in the existing sidebar overlay rather than as horizontal content tabs.
Settings destinations take over the main pane directly and do not reuse the session chat title bar;
each destination's own heading is the page heading.

The composer footer is the sole compact runtime-information surface outside Settings. It is
rendered inside the composer border at the lower left. The sidebar,
title bar, and other application chrome do not duplicate model identity, residency, allocation, or
context information.

CLI and web share the pure five-axis local-model comparison profile: intelligence, speed,
speculation, memory efficiency, and accuracy. Each client owns its renderer, so terminal cells and
browser SVG remain separate presentations of the same model evidence.

Install, update, cancellation, failure dismissal, removal, and warm load use the canonical model
ID. Selection assigns that same model ID to a slot. Active Stop asks ACN to resolve and retire the
current native instance; slot Stop addresses the slot. Native download and instance IDs remain
private. Long-running progress comes from canonical service queries. Mutation pending and failure
describe the command separately; command acknowledgement does not fabricate acquisition or residency
progress. Mutation synchronization refreshes the already-committed catalog snapshot.

Chat submission requires a selected local model. When no model is selected, the composer routes the
user to Models in Settings instead of discarding the attempt. A selected model does not need to be
resident before submission: the inference request automatically acquires it through ICN's residency
coordinator. Native Instance events and opt-in request progress provide loading status while the
message waits. The client must not treat a selected unloaded, loading, stopping, or failed model as
though no model were selected. If model-slot state is temporarily unavailable, the
client likewise must not infer that selection is absent.
While acquisition is requested or loading, the work-status surface gives model loading priority
over the generic waiting detail and renders `Loading model`, a spinner, and authoritative progress
when available.

The composer footer follows the CLI's runtime-information structure while presenting model identity
and reasoning effort as one compact configuration control. It also presents resident memory and
context usage with percentage. Residency is not rendered as a dot or readiness label in the
composer. The combined control has a Phosphor rocket-launch icon, model name, reasoning label, and
caret, with no separator between the model and reasoning text. It has no persistent border,
underline, or hover text-color shift; hover is communicated with a restrained background change.
Its upward root menu contains Model and, when the selected or drafted model supports reasoning,
Thinking rows whose submenus open to the right by default. Models without configurable reasoning do
not show a reasoning label or menu row. A reasoning-capable model whose selected effort is `none`
shows `None` rather than hiding the control.
Choosing a model advances directly to that model's supported thinking levels. The model and
reasoning effort are committed together only after the sequence is complete; dismissing an
incomplete sequence discards its presentation draft. A user may open Thinking directly to change
only the current model's effort. The menu and submenus are keyboard operable, and resident memory
routes to Hardware.

A selected model retains the ordinary foreground color regardless of residency. Loading is
communicated by the work-status activity instead of muting model identity. The combined menu also
reflects authoritative inventory and discovery availability: without a selection its trigger
identifies loading, and with a selection it preserves the selected identity while its menu reports
that additional choices are still loading. Query failure is distinct from both loading and a
successfully loaded empty list. Model and reasoning text use the slate hierarchy rather than a
separate semantic accent color.

Context usage is normally a compact circular meter rather than persistent text. Its arc is blue
below 70 percent usage, orange from 70 through 89 percent, and red at 90 percent or above. Hover or
keyboard focus reveals a fixed three-line tooltip containing the state label, token count, and
percent remaining. While the authoritative root actor context reports compaction, the label becomes
`Compacting...` and the same-length arc turns violet and rotates counterclockwise; the client does
not infer compaction from token movement or fabricate a reduced usage value. The composer keeps the
model control at the right edge beside Send, with a deliberate gap, and places context immediately
to its left. Context is absent from a truly empty chat and appears once the root timeline contains a
message, including an optimistically accepted user message.

## Appearance

Web and Electron share one renderer-owned appearance preference: `system`, `light`, or `dark`.
`system` is the default and follows `prefers-color-scheme`, including live operating-system changes.
An explicit light or dark choice is persisted in renderer-local storage. This preference is local
presentation state and does not belong in ACN, SDK, or client-common.

The resolved appearance sets the document theme selector and selects a matching
syntax-highlighting theme. Components express appearance with direct Tailwind palette utilities and
`dark:` variants. The Tailwind color namespace contains only the canonical Magnitude palette plus
explicit black, white, and transparent values; components do not introduce raw colors, arbitrary
palette-color utilities, runtime palette mixing, or a parallel semantic color-token layer. Literal
black alpha is reserved for shadows and overlays. Repeated visual behavior is shared through
meaningful React components rather than CSS component classes.

Web and Electron bundle their typography rather than depending on host-installed fonts. Inter is
the body and interface family, including form controls. Martian Mono is the heading family. The
ordinary monospace stack remains reserved for code and technical data rather than page headings.

Static presentation uses Tailwind utilities. Inline styles are reserved for values derived from
runtime data, such as measured dimensions, progress values, and SVG geometry, or for third-party
renderer output that cannot consume classes. The headless CLI has no terminal appearance detector,
interactive theme, or parallel onboarding presentation.

## Conformance

- Browser and desktop use canonical shared model observations and recover by refetching service state.
- Browser startup observes SDK lifecycle; desktop observes its owned application lifecycle.
- No web model-management component reconstructs the deprecated candidate/offering/download model.
- Non-local catalog entries are never rendered, selected, assigned, or counted as ready.
- Every long-running lifecycle is rendered from server state; mutations cover command admission.
- Settings is a first-class application surface with Models, Catalog, and Hardware destinations.
- Wide and narrow layouts preserve access to every model-management view and action.
- Footer model and reasoning choices remain available without navigating away from the chat.
- Catalog keeps the selected model's primary action visible while its evidence scrolls, labels
  radar axes directly, and presents license/source metadata
  below it, and does not repeat radar evidence in candidate rows or metric tiles. The chart remains
  fully contained without horizontal scrolling. Search,
  the installed-model filter, and the labeled sort control remain visually distinct. Its
  two-pane layout is one unified browser surface, consumes the available Settings height without
  viewport-height arithmetic, and gives each pane its own overflow so the detail surface neither
  clips nor leaves a false bottom gap.
- Slash commands and host menu actions route to the corresponding web-native surface.
- System appearance is the default, explicit overrides persist locally, and code highlighting tracks
  the resolved theme.
- Browser and Electron builds compile the same direct Tailwind treatments from the approved
  Magnitude palette; unauthorized colors and handwritten feature CSS are mechanically rejected.
