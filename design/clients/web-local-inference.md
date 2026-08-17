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
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - packages/client-common/src/local-models/**
  - packages/client-common/src/model-slots/**
---

# Web local inference and appearance

The browser product and Electron renderer expose one local-only model experience. Electron hosts
the browser renderer and supplies desktop capabilities and daemon bootstrap; it does not own a
separate model-management or appearance interface.

## Authority and boundaries

`LocalModelsState` is the server-owned product projection for inventory, catalog membership,
acquisition, update, serving configuration, assessment, availability, and recommendations. Clients
must not join older parallel package, candidate, and offering collections to reconstruct that
product state. `ModelSlotsState` owns durable selection, availability, residency, canonical actions,
favorites, and resident allocation. Hardware and onboarding completion likewise remain server
owned.

Web consumes those domains through the connection-scoped `LocalModels`, `ModelSlots`, and
`OnboardingModelSetup` client-common services. React hooks are adapters to those services. DOM
components may derive labels and layout, but do not construct services, cache server snapshots, or
infer compatibility, availability, readiness, progress, or command completion.

The web product presents only local models. Protocol support for another provider does not make it
a web product choice or readiness signal. Cloud login, account usage, subscription, connection,
and cloud-model surfaces are absent.

## Product flow

Startup presents daemon lifecycle, the shared onboarding setup view, or the ordinary application
shell as mutually exclusive states. Required onboarding prevents session preload and chat entry.
The shared setup service sequences install, assignment, load, cancellation, and onboarding
completion and publishes one server-derived presentation state. Web renders that state directly and
does not maintain a second workflow or correlate operations itself.

The ordinary shell contains a dedicated Model Center:

- Models presents selected slots, installed models, active transfers, failures, and residency.
- Catalog presents the unified assessed local catalog and recommendation evidence.
- Hardware presents server-reported topology and a labeled physical-memory breakdown alongside
  resident allocations. Internal admission thresholds are not exposed as end-user concepts.

The chat footer is the sole compact runtime-information surface outside Model Center. The sidebar,
title bar, and other application chrome do not duplicate model identity, residency, allocation, or
context information.

CLI and web share the pure five-axis local-model comparison profile: intelligence, speed,
speculation, memory efficiency, and accuracy. Each client owns its renderer, so terminal cells and
browser SVG remain separate presentations of the same model evidence.

Install and update use the model serving configuration ID. Cancellation and failure dismissal use
the exact server-issued download ID. Selection assigns the projected local provider-model identity
to a slot. Load and stop address the slot; the daemon owns the physical instance identity. Long
running state is always rendered from refreshed service queries, while mutation state represents
only local invocation admission.

Chat readiness requires an available local slot whose residency is `Ready`. Selection alone is not
readiness. When submission is unavailable, the composer routes the user to Model Center instead of
discarding the attempt.

The chat footer follows the CLI's runtime-information structure without combining independent
facts into the model label. It presents residency, model identity, reasoning effort, resident
memory, context usage with percentage, and working directory in that order. Model identity routes
to Models, resident memory routes to Hardware, and reasoning choices expand inline. Interactive
labels are underlined only while hovered; reasoning retains its violet semantic color.

## Appearance

Web and Electron share one renderer-owned appearance preference: `system`, `light`, or `dark`.
`system` is the default and follows `prefers-color-scheme`, including live operating-system changes.
An explicit light or dark choice is persisted in renderer-local storage. This preference is local
presentation state and does not belong in ACN, SDK, or client-common.

The resolved appearance selects semantic CSS variables and a matching syntax-highlighting theme.
Components consume semantic variables and must not branch on appearance or introduce raw
mode-specific colors. The terminal appearance detector and `CliTheme` remain CLI-owned because
terminal palette discovery and transparent terminal backgrounds are not browser semantics.

## Conformance

- Browser and Electron render the same React application and recover by refetching service state.
- No web model-management component reconstructs the deprecated candidate/offering/download model.
- Non-local catalog entries are never rendered, selected, assigned, or counted as ready.
- Every long-running lifecycle is rendered from server state; mutations cover command admission.
- Model Center is a first-class application surface rather than a Settings subsection.
- Wide and narrow layouts preserve access to every model-management view and action.
- Catalog keeps the selected model's primary action visible while its evidence scrolls, labels
  radar axes directly, and does not repeat radar evidence in candidate rows or metric tiles.
- Slash commands and host menu actions route to the corresponding web-native surface.
- System appearance is the default, explicit overrides persist locally, and code highlighting tracks
  the resolved theme.
