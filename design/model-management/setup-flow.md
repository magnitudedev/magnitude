---
applies_to:
  - packages/acn-protocol/src/boundary/onboarding.ts
  - packages/acn/src/onboarding/**
  - packages/client-common/src/onboarding/**
  - packages/client-common/src/desktop/onboarding*
  - packages/client-common/src/state/client-services.ts
  - desktop/src/renderer.tsx
---

# Desktop onboarding

The desktop is the only guided onboarding surface. The headless CLI invokes model and connection
capabilities independently. Web and external harnesses do not host a second onboarding flow.

ACN owns one durable, monotonic Incomplete → Complete fact. Explicit Finish or Skip completes it
idempotently. Completion records dismissal of first-use guidance, not proof that a model is installed,
loaded, or connected. Existing completion is preserved.

First use follows Discover and Connections inside the ordinary desktop shell. It reuses the curated
catalog, hardware assessment, and shared ranking preference. There is no preparation gate waiting
for every assessment: pending and unavailable evidence remains explicit and only fitting assessed
models are eligible. It uses the same slate/blue visual identity and appearance preference as the app.

The connection-scoped desktop workflow retains only interaction state and exact selected catalog ID.
Canonical model and onboarding queries remain the authorities. A selected model is synchronized;
observed completed installation permits a direct load command for that same model; observed ready
residency permits optional connection setup. There is no primary-slot assignment, Magnitude harness,
external process launch, or hosted return mode. Finish/Skip is explicit and is not automatically
inferred from a download, load, or connection acknowledgement.

One model setup may run at a time. The workflow's worker belongs to the client scope, survives page
navigation, and ends when that scope closes. Closing the window retains that scope. Cancelling a
download or stopping the model acts through the ordinary model commands; setup observes the result
and does not advance. Failures retain the selected identity and actionable explanation. Retry starts
a new selection and never silently substitutes another model. Finish/Skip cannot race active work.

Progress comes from the exact selected model's acquisition and residency observations, not invented
percentages or mutation history. The public load command does not return an instance identity;
readiness is therefore a current observation of that exact model, not a certificate of a particular
native instance. Removing or stopping it does not manufacture successful setup.

Background startup never opens the window or triggers onboarding mutations. Only a user selection
admits setup work; mounting the client service is observational. CLI commands remain independent of
the completion flag. Connections write configuration and required artifacts through the shared host
connector and never launch the selected tool.

## Acceptance

- Existing completed users see no first-use guidance; fresh users can explicitly Skip.
- Background first launch leaves the window hidden.
- Selection preserves exact catalog identity across ranking and catalog refreshes.
- Download acknowledgement alone never starts the connection step.
- Cancellation, failed installation, stopped loading, and observation failure do not advance setup.
- Navigation does not lose progress or admit a second worker.
- Completion persists across application recreation and does not alter model residency.
- No CLI/web/hosted onboarding or internal Magnitude harness path remains in the flow.
