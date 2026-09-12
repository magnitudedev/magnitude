---
applies_to:
  - packages/client-common/src/desktop/**
  - packages/client-common/src/state/client-services.ts
  - desktop/src/renderer.tsx
---

# Getting started in the desktop

Discover, Catalog, My Models and Connections are the complete getting-started experience. First
launch uses the same actions and model state as subsequent launches. There is no setup wizard,
completion flag, Finish/Skip action, or setup-specific tray message.

Download acquires the selected model. Load is a separate explicit model command; downloads never
automatically load a model because the user is new. Connections configure external tools using the
ordinary model selection policy and never launch them. Empty states may explain these actions,
but do not create a second workflow or persist onboarding completion.

Background startup leaves the window hidden. Reading state never installs or loads a model. The
CLI remains headless; there is no terminal onboarding or internal Magnitude harness.

Acceptance: a fresh profile and an existing profile expose identical model actions, including
after relaunch; no setup banner, completion RPC or setup worker exists. Existing unrelated model,
configuration and user data remain untouched.
