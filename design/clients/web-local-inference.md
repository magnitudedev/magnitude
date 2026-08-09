---
applies_to:
  - web/src/app.tsx
  - web/src/commands/**
  - web/src/components/composer.tsx
  - web/src/components/local-model-onboarding.tsx
  - web/src/components/model-center.tsx
  - web/src/hooks/use-menu-actions.ts
  - web/src/state/web-atoms.ts
  - packages/client-common/src/hooks/onboarding-model-view.ts
  - packages/client-common/src/hooks/use-local-inference-state.ts
  - packages/client-common/src/utils/local-inference-selections.ts
---

# Web local inference

The browser product and the Electron renderer expose one local-only model experience. Electron
hosts the browser renderer and supplies desktop capabilities and daemon bootstrap; it does not own
a separate model-management interface.

## Authority and boundaries

Hardware, model inventory, assessment, recommendations, downloads, provider offerings, slots,
instances, allocations, actions, and onboarding completion are authoritative ACN state. Clients
observe these domains through independent mirrored queries, derive presentation from successful
values, and issue exact commands through shared client actions. A renderer never copies a snapshot,
reconstructs progress, or infers compatibility, availability, readiness, or command completion.

The web product presents only local provider offerings. Protocol support for another provider does
not make it a web product choice or readiness signal. Cloud login, account usage, subscription,
connection, and cloud-model surfaces are absent.

## Product flow

Startup presents daemon lifecycle, required local-model onboarding, or the ordinary application
shell as mutually exclusive states. Required onboarding prevents session preload and chat entry.
Onboarding uses the shared finite setup operation and derives each visible phase from its exact
submitted identities plus authoritative mirrors.

The ordinary shell contains a dedicated Model Center with Models, Catalog, and Hardware views:

- Models presents selected slots, installed offerings, downloads, failures, and instance lifecycle.
- Catalog presents assessed local candidates and server recommendation evidence.
- Hardware presents server-reported topology, capacity, reserves, and resident allocations.

Catalog selection resolves or creates the offering for the exact assessed configuration and assigns
its provider-model identity to the requested slot. Download cancellation and instance stopping use
the exact admitted attempt or instance identities.

Chat displays selected-local-model and canonical-instance state. Selection alone is not readiness.
When submission is unavailable, the composer remains actionable and routes the user to Model Center
instead of silently discarding the attempt.

## Client sharing

Clients share only renderer-neutral semantics: local choice construction, onboarding phase
derivation, the finite configuration-to-offering-to-slot operation, mirror hooks, and exact mutation
actions. React DOM and OpenTUI retain independent layout, copy, formatting, focus, and navigation.

## Conformance

- Browser and Electron render the same React application and recover by refetching mirrors.
- Non-local catalog entries are never rendered, selected, assigned, or counted as ready.
- Every long-running lifecycle is rendered from server state; mutation state covers command admission
  only.
- Model Center is a first-class application surface rather than a Settings subsection.
- Wide and narrow layouts preserve access to every model-management view and action.
- Slash commands and host menu actions route to the corresponding web-native surface.
