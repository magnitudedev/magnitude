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
Quick permits an explicit harness override without changing its selected case IDs. Broader
profiles retain all mandatory harnesses; narrowing them requires an explicit custom selection.
The planner and worker use the same harness selection policy.

## Responsibilities

The coordinator durably records admission, work, attempts, leases, events and results.
Database migrations finish before HTTP admission starts. Scheduler and reconciliation loops run
under the coordinator's scope; unexpected loop defects terminate the service rather than silently
leaving a live API without its worker or cleanup loop. Allocators
own Azure, Namespace, office Spark or local-machine leases. Transports execute bounded worker
commands. Scenario drivers interact with the installed application, endpoint, actual harness or
bundled CLI. No mocked model response, developer binary or hosted provider can satisfy generation.

The nine suites are package, install, app, endpoint, harness, recovery, CLI, update and uninstall.
Pi, OpenCode and Hermes are the initial harnesses. Configuration must be produced through the
product's Connections surface. Bundled CLI connection acceptance additionally exercises
add, sync, removal and reconnection, inspecting the actual harness configuration after each
mutation. Command success alone cannot establish endpoint correctness or preservation of
unrelated provider settings. Endpoint and harness tests attest the requested backend/device;
CPU fallback fails a Metal/CUDA target. Host identity is collected from native OS and device
interfaces before test execution; a Windows Server build cannot qualify a Windows client target.
Unavailable GPU memory totals remain unknown, not fabricated. Device discovery alone cannot
qualify generation. Generation checks output/protocol/tool behavior, not speed.

## Ownership and recovery

Authentication maps credentials to server-owned identities and permissions. Developer identity
requires a tenant-specific, signature-verified Entra v2 access token for the lab API, the delegated
Lab.Access scope and an explicitly allowed immutable user object ID. Azure subscription roles
alone do not grant lab access. GitHub OIDC verifies
signature, issuer, audience, lifetime and an immutable repository/owner allowlist. Each workflow
run attempt has a separate owner and always receives untrusted CI permissions; token or request
claims cannot elevate it. Clients renew short-lived CI tokens before later API requests;
credential renewal never replays an upload or mutation. Pull-request-target workflows are not admitted by this policy.
Admission is idempotent and reserves a bounded budget. Durable transactional claims and monotonic
attempt fences prevent stale workers from committing results. Worker invocation binds the assignment and attempt fence; returned case membership and evidence
hashes are verified before acceptance. Transfers expose only the owner-authorized input graph.
Workers receive only scoped run credentials when needed; untrusted source never receives provider credentials or office-network access.
Worker credentials identify one admitted target attempt. Only their digests are persisted.
Every use checks the current work fence, claim expiry and run state, so cancellation,
completion or reassignment invalidates access. Issuing another credential for the same
attempt is rejected rather than silently replacing a credential already delivered to a guest.
Worker input access is limited to the assigned manifest and its referenced content objects.
Knowledge of a digest, including one uploaded by the same owner for another purpose, grants
no access outside that graph. Manifest size and integrity are checked before authorizing
referenced content.
The immutable manifest graph may be cached within bounded memory; live worker authorization
is never cached. Failed graph reads are evicted so later requests can recover from an
infrastructure failure without replaying a test assertion.
Worker result receipts are immutable per attempt and require exact selected-case membership.
Evidence references must match verified uploads for that attempt. Repeated identical delivery
is idempotent; a changed reply is rejected. Receipt insertion and live authority validation
share one database transaction, preventing cancellation or reassignment from racing acceptance.
Evidence uploads reserve their declared bytes against an attempt budget before consuming
the request. Uploading and verified objects are distinct states. Hash/length verification
and a fresh authority check precede publication; incomplete uploads cannot satisfy a result.
Concurrent reservations count toward the same budget, and failed uploads release only their
own reservation. Expired reservations can be reclaimed without granting evidence authority.
Guest HTTP clients use one HTTPS coordinator origin (loopback HTTP is permitted for local
tests), reject redirects and verify downloaded content addresses. Transport errors never
automatically repeat native test execution. Assignment, transfer and reply sizes and waits
are bounded independently of the overall attempt deadline.
An outward guest claims a fresh local workspace before downloading or executing inputs.
An existing workspace cannot trigger another execution. The guest saves its validated reply
before delivery, deduplicates evidence by content identity and monitors live assignment
authority while working. Lost authority or the attempt deadline interrupts execution and
its owned cleanup scope; transport failures do not silently rerun tests.
Explicit delivery recovery requires the saved invocation to match the entire live assignment
and validates the saved reply, evidence lengths and hashes again. It never acquires the native
executor or downloads execution inputs. Missing, malformed or oversized saved documents fail;
they cannot fall back to another execution. Recovery retains the original attempt deadline and
authority checks, and cannot resurrect a revoked or completed attempt.
The outward runner validates allocation ownership and trust before issuing a guest credential,
delivers it through a provider bootstrap, and waits for the immutable receipt within the
allocation deadline. Credential revocation runs on success, failure, timeout and cancellation.
Revocation failures remain separate cleanup errors when a valid test result exists; the
scheduler retains responsibility for releasing the allocated machine.
Provider bootstraps verify the resource's exact lease identity before delivering authority.
Credentials must not appear in command arguments, script text or returned provider errors;
temporary delivery files are private and scoped. Guest launch uses the intended application
user and a qualified display environment. Service-session execution cannot qualify an
interactive desktop test, and accepting a launch request alone never establishes a test pass.
Spark is opt-in for trusted source, exclusive within the lab and subject to a busy-device check.

Installed-package ownership follows explicit present/absent transitions. Removal updates
ownership only after the native uninstaller succeeds; reinstall acquires new ownership.
Final cleanup removes only the currently owned installation. Dangling package launcher
symlinks count as removal failures even when their targets no longer exist.
User-data retention compares the stopped application's complete isolated profile before and
after native removal, including file contents, directories and symbolic links without following
links outside the profile. Evidence contains hashes rather than file contents. Reinstall must
consume the same candidate and retain an application setting across the removal boundary.

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
Update controls expose rendered transfer state and candidate version through stable identities.
The driver uses the normal Settings actions and surfaces failed or unavailable updates promptly.
A renderer interaction test alone cannot qualify installation, trust or retained-data acceptance;
those require an old/new package pair and observation of the resulting installed application.
Settings persistence is checked across actual application process restarts using the same
isolated profile; a page reload cannot satisfy it. Each launch preserves its own trace and
process log, and the previous process is released before the next launches. Application quit
must traverse normal shutdown and terminate the main process with a successful exit code;
window hiding and forced cleanup cannot satisfy quit acceptance. Repeated service starts
retain the native application PID, service PID and service instance identity. After a
restart the previous owning service must have exited and the replacement must have
a new identity; tests never terminate a leftover process to manufacture acceptance.
CLI interruption acceptance first observes a connection to the deliberately stalled owned
service, then interrupts the CLI and requires the expected interruption exit. Early process
exit cannot satisfy this case. The service is resumed on every exit path, and afterward the
same application and service identities must remain usable. Unsupported native interruption
mechanisms block qualification rather than substituting forced termination.
Error scenarios preserve malformed harness files and restore the original fixture bytes
even after failed assertions. Visible error messages and file-specific repair guidance
may occupy separate elements. Service failure injection owns only an isolated loopback
listener; recovery requires successful startup after that listener is released.
Screenshots are diagnostics rather than visual acceptance baselines. A presentation perturbation
probe verifies that copy and layout changes do not break functional paths.
JSON and JUnit reports derive from the same admitted plan and completed result.
JUnit distinguishes product failures from infrastructure errors; missing, blocked,
cancelled or duplicate selected results and cleanup failures cannot become green checks.
Finalized UI traces and command logs are published as content-addressed evidence before worker
removal. Evidence export failures remain visible without replacing the original case outcome.

Blocked prerequisites propagate without hiding the original product failure. Pass means every
required selected case passed and cleanup completed. Tests must demonstrate corrupt-input rejection,
source fidelity, cancellation, stale-worker fencing, allocation ambiguity recovery, backend mismatch
rejection and exact package consumption. Execution on one target never qualifies another.

Linux coverage is limited to Ubuntu, Debian, Fedora and Red Hat using existing DEB/RPM artifacts.
SUSE, Arch, Omarchy and Alpine packaging are future work. Provider credits and quotas do not confer
Windows licensing, signing identities or GPU availability; these are independently observed gates.
