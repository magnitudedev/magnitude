---
applies_to:
  - packages/daemon-management/src/desktop-native/owned-service.ts
  - packages/acn-protocol/src/acn-identity.ts
  - packages/acn-protocol/src/acn-revision.ts
  - packages/acn/src/server.ts
  - packages/acn/src/icn/**
  - packages/version/scripts/generate-version.ts
  - packages/release/release-plan.json
  - packages/release/src/release-plan.ts
  - desktop/src/main.ts
  - web/scripts/dev-server.ts
---

# Desktop-owned service supervision

The desktop application is the sole production service owner. Login, CLI commands, and supported
harness starters launch or contact that application. Only its privileged main process launches ACN;
ACN privately owns ICN and workers. No client independently downloads, adopts, elects, replaces, or
retains an ACN process.
The web development host uses the same application launcher with an isolated development profile
and proxies that profile's service endpoint. Closing the web host or cancelling its startup response
does not quit an already admitted desktop owner.

A kernel-held application lock excludes concurrent owners without killing a stalled predecessor.
A losing launch forwards background Ensure or explicit Open intent over per-user local IPC, or
returns a bounded owner-unavailable result. Background demand never shows or focuses a window.
The SDK remains portable and receives a starter capability rather than OS process authority.

## Child supervision

The parent retains the raw child handle from creation, then validates exact process identity before
sending Start over its inherited control channel. The child installs native parent-loss protection
before application initialization and keeps it after Ready. Child health derives from the existing
ACN lifecycle. Readiness requires matching instance and RPC identity, not merely a PID or HTTP 200.

The parent owns bounded crash recovery. Every attempt first retires the predecessor's known process
tree. Unproven cleanup, PID reuse, an occupied endpoint, or observational failure permits neither
adoption nor a replacement. Diagnostic output is bounded and independent of control framing.

Shutdown is single-flight: reject new demand, cancel retries, request child Shutdown, wait, escalate
against the exact owned group/job, and prove absence. Root exit alone does not prove descendant
exit. The application releases its lock and tray only after shutdown, or an explicit force-quit
that truthfully reports unproven cleanup. Requester cancellation cannot abandon teardown.

Unix lifetime guards remain independent of JavaScript; separately grouped children require their
own protection. Windows uses job containment assigned at process creation. Native platform testing
is required in addition to unit tests.

## Compatibility and migration

Exact SDK RPC-version and instance fencing remain. Application updates own replacement of their
matched bundled service; an older CLI cannot downgrade a live app's child. Legacy SQLite records
and daemon jobs are consulted only by isolated one-time migration, which validates exact identity,
retires the old tree, preserves data/login preference, and removes obsolete supervision. They are
never normal runtime ownership authority.

Existing SDK connections reconnect only to an existing owner. Full Quit cannot be undone by stale
subscriptions or background retries. Fresh explicit demand may launch a new application lifetime.

## Conformance

- One application owner, one service child, one private inference tree.
- No database election, convergence, admitted detachment, or direct production daemon fallback.
- CLI/login/harness starts preserve background intent, even during concurrent launches.
- Window close preserves the tray/service; full Quit retires the owned tree.
- A failed or unresponsive owner is never implicitly killed by a contender.
- Replacement requires exact predecessor cleanup and fresh RPC admission.
