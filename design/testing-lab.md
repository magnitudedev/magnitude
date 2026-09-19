---
applies_to:
  - packages/testing-lab/**
  - scripts/lab.ts
  - .github/workflows/testing-lab.yml
---

# Testing lab

The lab validates final application packages and their user-visible behavior. It owns test
orchestration, disposable environments, fixtures and evidence; release owns compilation,
packaging and trust. Product runtimes never import the lab. Benchmarks are not acceptance gates.

## Inputs and execution

A run binds an immutable source snapshot or existing artifact manifest, explicit target and case
selection, isolation mode and resource limits. Source includes tracked changes, nonignored new
files and recursive submodule working trees. Concurrent mutation invalidates a snapshot. Every
download and extracted file is integrity checked; extraction cannot escape its owned directory.
Existing artifacts are copied into content-addressed storage and verified against their declared
hashes and lengths before submission. Subsequent edits to local packages cannot alter an admitted
input. Existing artifacts are never rebuilt or relabelled as a successful compilation. Source execution
extracts the admitted snapshot into a fresh workspace, installs frozen dependencies, and records
compilation and final packaging independently. A failed compile blocks packaging and consumption.
Build receipts bind the snapshot digest, base commit and native host; final manifests and every
package byte are verified before the installer can consume them. A base commit alone never
identifies dirty source. Producer reuse across targets and separate consumer allocation remain
implementation work; same-machine source probes do not qualify the clean remote consumer gate.

`iterate` may reuse an owner-scoped lease and build cache. `verify` uses clean source, build output
and consumer state. Neither mode may touch a developer's normal application data. Local execution
requires an isolated profile, and privileged install/uninstall is restricted to disposable hosts.

`quick`, `pr`, `full` and `release` select cases and targets. Target expansion is explicit, including
unqualified targets. Missing capacity or credentials blocks a case; it never removes it or makes
it pass. Release acceptance requires final production-signed artifacts and complete coverage.

## Responsibilities

The coordinator durably records admission, work, attempts, leases, events and results. Allocators
own Azure, Namespace, office Spark or local-machine leases. Transports execute bounded worker
commands. Scenario drivers interact with the installed application, endpoint, actual harness or
bundled CLI. No mocked model response, developer binary or hosted provider can satisfy generation.

The nine suites are package, install, app, endpoint, harness, recovery, CLI, update and uninstall.
Pi, OpenCode and Hermes are the initial harnesses. Configuration must be produced through the
product's Connections surface. Endpoint and harness tests attest the requested backend/device;
CPU fallback fails a Metal/CUDA target. Host identity is collected from native OS and device
interfaces before test execution; a Windows Server build cannot qualify a Windows client target.
Unavailable GPU memory totals remain unknown, not fabricated. Device discovery alone cannot
qualify generation. Generation checks output/protocol/tool behavior, not speed.

## Ownership and recovery

Admission is idempotent and reserves a bounded budget. Durable transactional claims and monotonic
attempt fences prevent stale workers from committing results. Worker invocation binds the assignment and attempt fence; returned case membership and evidence
hashes are verified before acceptance. Transfers expose only the owner-authorized input graph.
Workers receive only scoped run credentials when needed; untrusted source never receives provider credentials or office-network access.
Spark is opt-in for trusted source, exclusive within the lab and subject to a busy-device check.

Every rented resource is tagged with lab/run/lease identity and an absolute expiry. Cancellation,
timeouts and failures release owned resources; reconciliation also inventories providers to find
allocations lost between provisioning and persistence. Cleanup failures remain visible independently
of test results. No test assertion is automatically retried. Infrastructure retry preserves the
original attempt and evidence. Cleanup deadlines remain effective inside finalizers, and forced
termination is limited to processes owned by the attempt.

## Evidence and acceptance

Each case records its input identity, observed hardware, outcome, diagnostics and evidence paths.
Installed package identity is verified against native binary headers and the running desktop,
service and CLI versions. Artifact names alone cannot qualify architecture or version agreement.
UI drivers address stable action and entity identities rather than copy, colors, geometry, or
DOM position. UI redesign preserves those identities; changed workflows are centralized in the
driver. Waits observe semantic state, while endpoint and harness behavior prove actual operation.
Screenshots are diagnostics rather than visual acceptance baselines. A presentation perturbation
probe verifies that copy and layout changes do not break functional paths.

Blocked prerequisites propagate without hiding the original product failure. Pass means every
required selected case passed and cleanup completed. Tests must demonstrate corrupt-input rejection,
source fidelity, cancellation, stale-worker fencing, allocation ambiguity recovery, backend mismatch
rejection and exact package consumption. Execution on one target never qualifies another.

Linux coverage is limited to Ubuntu, Debian, Fedora and Red Hat using existing DEB/RPM artifacts.
SUSE, Arch, Omarchy and Alpine packaging are future work. Provider credits and quotas do not confer
Windows licensing, signing identities or GPU availability; these are independently observed gates.
