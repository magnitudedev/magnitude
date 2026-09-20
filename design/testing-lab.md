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
Source graph admission checks owner-scoped object lengths in bounded batches, including every
reference when multiple files share a digest. Azure object transport uses a shared HTTP client
and a cached, renewable Entra token. Uploads are hash checked before conditional publication;
downloads bind an observed ETag, enforce length limits and verify the content hash. Redirects
are disabled, and authentication failures never become missing-object results.
Source candidates compile the release-owned CPU base and the selected host's backend pack alongside
the desktop. CPU targets need no additional pack; Metal and CUDA targets require their matching
pack. Compilation receipts bind the selected backend and native build identity. Final packaging
archives those exact compiled inputs and admits desktop, base and pack bytes together; it cannot
silently substitute a published inference runtime for unpublished source changes.
The manifest also includes the canonical archive of the bundled service, whose size is consumed
by ordinary runtime startup. Package admission validates that startup's bundle metadata is complete.
Build receipts bind the snapshot digest, base commit and native host; final manifests and every
package byte are verified before the installer can consume them. A base commit alone never
identifies dirty source. Source runs share one producer for each native artifact host and backend within that run.
A producer uses a canonical native CPU build host and a separately selected backend toolchain;
CUDA compilation does not implicitly allocate a GPU or use Spark. Build and test work have distinct
identities, fenced attempts, credentials and allocations even when they use the same target.
Consumers cannot start until the producer has published a complete verified package graph and its
allocation has been released. Publication binds the source digest, commit, host, backend and actual
producer lease. Build failures block dependent tests without allocating consumer machines.
Consumers receive only the admitted packages and selected update baseline, never source or an
inline compiler. Final target reports include the actual shared producer outcomes and receipt,
then that consumer's outcomes, exactly once per originally selected case. Producer cleanup errors
are aggregated once and prevent dependent execution. Historical completed reports survive the
work-identity migration; active old-protocol attempts and leases must drain before migration.
An update run may additionally bind a previous-release artifact manifest. Its packages are
frozen and verified like candidate artifacts, registered under the same owner and included in
the immutable assignment. Admission requires both inputs. Worker transfer authority includes
exactly their manifest graphs; unrelated uploaded artifacts remain inaccessible. Supplying a
baseline alone does not establish version ordering, update trust or a successful installation.
Runtime release fixtures preserve the admitted manifest and expose only its selected host's native
base and backend archives through ordinary release URLs. Verified private copies and a random-route
loopback listener share the run scope. Publication rejects missing bases, unsafe names and corrupt
bytes; delivery supports bounded byte ranges without redirecting to external or development files.
Successful fixture transport alone does not qualify compilation, extraction or backend execution.
Candidate desktop and bundled CLI sessions share their candidate runtime origin. An update baseline
uses its own validated release manifest and separately scoped origin; it cannot inherit the candidate
origin. Worker acceptance removes ambient development-installation overrides. An app-only input
retains ordinary released-runtime acquisition and does not qualify unpublished inference changes.

`iterate` may reuse an owner-scoped lease and build cache. `verify` uses clean source, build output
and consumer state. Neither mode may touch a developer's normal application data. Local execution
requires an isolated profile, and privileged install/uninstall is restricted to disposable hosts.
Desktop launches explicitly distinguish an isolated profile from an installed OS-user context.
The driver must not silently rewrite profile or endpoint settings: isolated launch configuration
must agree with the environment shared by the bundled CLI. Installed-user mode requires a
separately supplied qualified disposable-user capability, uses normal product data and endpoint
policy, and rejects development profile or control-directory overrides. A HOME substitution is
not proof of disposable OS-user authority. Existing isolated probes cannot qualify native login
registration; provider/user qualification and installed-mode orchestration are required separately.
The shared worker derives application, CLI, endpoint, harness and retained-data paths from one
context and records its mode and nonsecret paths in case evidence. A native guest context uses
the actual OS account, verifies non-root Unix identity and home ownership, and rejects preexisting
application data, including dangling links. Only coordinator-designated disposable cloud workers
can grant that context; shared-host runners reject a disposable claim even for trusted source.
Native Mac guest installation uses Applications so normal installed-app policies can be exercised.
These admission checks do not themselves qualify native login registration or update behavior.

`quick`, `pr`, `full` and `release` select cases and targets. Target expansion is explicit, including
unqualified targets. Missing capacity or credentials blocks a case; it never removes it or makes
it pass. Release acceptance requires final production-signed artifacts and complete coverage.
Quick permits an explicit harness override without changing its selected case IDs. Broader
profiles retain all mandatory harnesses; narrowing them requires an explicit custom selection.
The planner and worker use the same harness selection policy.

## Responsibilities

The coordinator durably records admission, work, attempts, leases, events and results.
The deployed coordinator keeps database and artifact state outside its container replica. Its
managed identity owns cloud allocation and storage access; that identity is never assigned to
workers. HTTPS ingress exposes the authenticated API, and the database uses private networking
and verified TLS. Infrastructure and runtime-image references are explicit deployment inputs.
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
mutation. Hermes first-run qualification explicitly selects its default model using the bundled
CLI after UI connection; this tests the existing user action without changing the UI connection
policy. Its worker installation pins the upstream source and dependencies, including its native
command scanner, so scanner startup diagnostics cannot masquerade as JSON generation events.
Command success alone cannot establish endpoint correctness or preservation of
unrelated provider settings. Endpoint and harness tests attest the requested backend/device;
CPU fallback fails a Metal/CUDA target. Host identity is collected from native OS and device
interfaces before test execution; a Windows Server build cannot qualify a Windows client target.
Unavailable GPU memory totals remain unknown, not fabricated. Device discovery alone cannot
qualify generation. Generation checks output/protocol/tool behavior, not speed. OpenCode CLI streaming acceptance
uses an observation-only hook on the pinned harness's native text-part updates and deltas.
An empty initial part, its deltas and its timestamped completion establish the streaming lifecycle.
The observed session and assistant identities must match the native exported transcript, and
concatenated deltas must match both completed CLI output and persisted text. Missing, duplicate,
truncated, foreign or unfinished lifecycle evidence fails qualification. Instrumentation may
record events but cannot change provider configuration, prompts, tool behavior or generated output.
Tool-result acceptance supplies a newly generated result that was absent from the original
prompt and arguments; echoing known input cannot prove that the model consumed the tool result.

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
Worker credentials identify one admitted build or test work attempt. Only their digests are persisted.
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
the request. Empty log files are valid zero-byte evidence objects and retain normal digest and
authority checks; an absent HTTP body with an explicit zero length is an empty stream. Uploading and verified objects are distinct states. Hash/length verification
and a fresh authority check precede publication; incomplete uploads cannot satisfy a result.
Concurrent reservations count toward the same budget, and failed uploads release only their
own reservation. Producers may upload package-sized objects up to 4 GiB within a 16 GiB attempt budget;
consumers retain a 256 MiB object and 4 GiB attempt evidence limit. A consumer cannot publish build
output. Complete package graphs must belong to the producing attempt before outward result receipt.
Expired reservations can be reclaimed without granting evidence authority.
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
Developers retrieve evidence through an authenticated run-scoped endpoint. It authorizes the
run owner and requires the digest to occur in the completed result; knowledge of a hash alone
cannot grant access through that endpoint. The CLI verifies bounded downloads
before publishing a local file and refuses to overwrite an existing destination.
The outward runner validates allocation ownership and trust before issuing a guest credential,
delivers it through a provider bootstrap, and waits for the immutable receipt within the
allocation deadline. Credential revocation runs on success, failure, timeout and cancellation.
Revocation failures remain separate cleanup errors when a valid test result exists; the
scheduler retains responsibility for releasing the allocated machine. Provider-native process
termination without a received result is an infrastructure failure, including exit zero. A final
receipt read resolves the race between delivery and exit observation. Read-only status observation
may retry transient transport failures; it never reruns native execution. Provider output is not
copied into public errors or allowed to expose the guest credential. Worker transport errors
identify their operation and resource path without headers. Server-side infrastructure failures
retain bounded, credential-redacted diagnostic logs; public error responses remain opaque.
Run progress exposes owner-authorized build/test stages, attempt counts, dependency identities
and current allocation states. The CLI emits stage changes while waiting; progress is not a test
result and cannot establish acceptance. Reconnecting to wait on an existing run never resubmits
its input or acquires another allocation; reports retain the original run identity.
Provider bootstraps verify the resource's exact lease identity before delivering authority.
Fresh Azure Linux workers may prepare their trusted runtime through administrator-pinned
cloud-init configuration or a pinned Ubuntu preparation recipe. The allocator verifies setup
bytes before creating resources and binds the preparation identity to the VM. A renewable recipe
pins archive identity and issues a fresh one-hour, blob-only read capability per allocation;
capability renewal does not alter the pinned preparation identity or grant provider credentials.
Native readiness requires completed cloud-init and the lab setup's final completion receipt.
Fatal cloud-init errors or a missing receipt fail admission. Recoverable platform warnings are
retained in detailed status and cannot substitute for the lab's successful completion receipt.
Azure provisioning success alone is not runtime readiness. Initialization receives no run or
provider credential; downloaded tooling has explicit length and digest checks. Candidate source
is delivered only after preparation. Preparation failures retain normal lease cleanup ownership. Before releasing a failed guest,
the coordinator stores bounded setup or terminal execution diagnostics in the artifact store, redacting credentials
and URL capabilities. Completed blocked cases reference those objects, so the run owner can
retrieve them after the guest is deleted. Diagnostic collection failure remains visible and
cannot suppress cleanup or turn preparation failure into success.
Run and provider credentials must not appear in command arguments, script text or returned provider errors;
temporary delivery files are private and scoped. Guest launch uses the intended application
user and a qualified display environment. Service-session execution cannot qualify an
interactive desktop test, and accepting a launch request alone never establishes a test pass.
Spark is opt-in for trusted source, exclusive within the lab and subject to a busy-device check.

Installed-package ownership follows explicit present/absent transitions. Removal updates
ownership only after the native uninstaller succeeds; reinstall acquires new ownership.
Final cleanup removes only the currently owned installation. Dangling package launcher
symlinks count as removal failures even when their targets no longer exist.
Fixture replacement serializes removal and installation under the same ownership gate.
Failed removal retains the previous owner; failed replacement installation leaves no owned
installation and retains the requested package for an explicit subsequent attempt. Cancellation
cannot discard ownership of a native installation that completed during the transition.
Installing a replacement directly is fixture preparation, never proof of application self-update.
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
Installation ownership verifies that the installer-selected CLI resolves to the exact bundled file
inside the installed package, including native launcher symlinks. Invoking its version and service
start commands must preserve the observed application PID, service PID and service instance.
This does not imply global shell registration where the platform does not install one.
Installed package identity is verified against native binary headers and the running desktop,
service and CLI versions. Artifact names alone cannot qualify architecture or version agreement.
Apple package trust verifies the installed application's sealed resources and every native file
in both the application and admitted runtime composition, across all signed architectures.
Development checks accept valid ad-hoc or certificate signatures as integrity evidence only;
they cannot claim production trust. Release checks require an explicitly configured expected
Apple team, Developer ID requirements, secure timestamps, the installed application's stapled
notarization ticket and a successful native Gatekeeper assessment. Missing publisher configuration
blocks release verification. Verification never re-signs a candidate or changes host trust.
Windows package trust inspects Authenticode on the admitted installer, every installed PE image,
and the admitted runtime composition. Required app/service/CLI, bridge, uninstaller and inference
entrypoints must appear in the native inventory. Development evidence records unsigned files
explicitly and rejects invalid signatures; it cannot claim publisher trust. Release verification
requires the configured expected publisher and SignTool for Magnitude-owned code, a valid timestamped
Authenticode result, and successful all-signature verification under the default authentication policy. Only known
Microsoft CRT filenames in the runtime library directory can use the explicit Microsoft publisher
exception. Paths enter PowerShell as literal-path environment data, never interpolated script code.
Other vendor code retains its vendor identity. Until its expected publisher or signed-container
provenance is independently verified, release trust remains blocked; an arbitrary valid signer
cannot satisfy it and vendor code must not be required to impersonate Magnitude.
Evidence distinguishes embedded Authenticode signatures from Windows catalog signatures and
unsigned code. A changed catalog-signed file can appear unsigned; that is not evidence of damaged
embedded-signature rejection. Negative native fixtures select an observed embedded signature and
change a covered PE section while preserving the signature structure.
These checks do not establish SmartScreen reputation or qualify a client OS from a Server probe.
Native dependency inspection preserves required, weak and delayed imports and platform loader
search paths. A missing or malformed tool report cannot become an empty successful graph.
Reading declarations does not establish dependency closure: acceptance must resolve the complete
owned graph from final application and runtime bytes without developer-machine search paths.
Mach-O resolution follows load-command ancestry and executable-relative paths, retains system
framework/shared-cache boundaries separately, and rejects missing owned imports, escaping symlinks
and ambient build-tool paths. A system boundary is not evidence that its shared-cache image was
independently inspected. Each executable and dynamically loaded package root requires coverage.
Native inventory deduplicates internal framework aliases, rejects escaping links, and reads the
selected universal-binary slice to assign executable contexts. Inspection tools use a scoped
plain-name alias when their filename grammar would reinterpret a package path; resolution still
uses the original installed loader context. Runtime inspection composes only integrity-verified
admitted base/selected-pack archives, requires matching native identities, and rejects path
collisions. Missing admitted runtime inputs block complete closure rather than borrowing a
developer installation. The fixture is scoped to inspection and cannot qualify runtime execution.
ELF owned search preserves the distinction between inherited RPATH and direct-only RUNPATH.
Origin-relative paths must remain inside the package; empty, ambient and unqualified token paths
fail. Search expansion alone cannot qualify resolved dependencies or system-library availability.
ELF traversal tracks inherited search context per object, verifies owned files and target
architecture, and records external dependencies only through an explicit system resolver that
verifies loader resolution and package ownership. A pathname dependency cannot fall back to an
OS basename lookup when its owned file is missing. Fixture resolvers do not qualify native hosts.
System resolution admits explicitly allowed names only, requires a unique architecture-matching
loader-cache record, checks canonical OS-directory ownership and ELF architecture, and records
the installed DEB/RPM owner. Ambiguous cache or hardware-capability choices block rather than
guessing loader precedence. Package ownership alone does not prove symbol-version compatibility.
Version requirements are matched by exact ABI identity against the resolved provider’s version
definitions, including owned dependencies. Missing or truncated version reports fail; a newer-looking
version number cannot substitute for an absent required identity.
Linux package inspection covers every installed native file and the admitted runtime composition.
Program headers must identify the architecture’s standard GNU loader when an interpreter is
present, and that loader must exist as an OS-directory ELF image of the correct architecture.
Only explicit desktop/system ABI names may cross the OS boundary; inference implementation and
CUDA toolkit libraries must remain owned. The NVIDIA driver library is admitted only for CUDA.
UI drivers address stable action and entity identities rather than copy, colors, geometry, or
DOM position. UI redesign preserves those identities; changed workflows are centralized in the
driver. Waits observe semantic state, while endpoint and harness behavior prove actual operation.
Connection error acceptance requires a nonempty rendered alert and visible affected-file guidance;
the driver opens a native disclosure when necessary rather than depending on its default expansion
state or label. Hidden configuration text alone cannot satisfy the assertion.
Update controls expose rendered transfer state and candidate version through stable identities.
The driver uses the normal Settings actions and surfaces failed or unavailable updates promptly.
A renderer interaction test alone cannot qualify installation, trust or retained-data acceptance;
those require an old/new package pair and observation of the resulting installed application.
The baseline case consumes both admitted artifact graphs before changing any installation. It
uses a separate profile and native control directory, verifies the previous desktop/service/CLI
payload identity, and proves persisted settings across a new application and service instance.
System package managers have one installation: baseline setup suspends the primary desktop,
temporarily replaces the package, closes the baseline app and restores the prior package ownership.
An originally absent installation remains lazy. Cleanup or restoration failure prevents later
cases from using uncertain native state. Direct package restoration is fixture management and
cannot qualify an application-driven update. Baseline traces and profile digests are retained.
Source producers whose consumers select update cases additionally build a same-source acceptance
pair through release-owned compilation and packaging. Each fixture has a complete version-matched
application/service/runtime artifact graph; the normal package remains separate. These fixtures
qualify updater functionality, not compatibility with historical released code. An explicit previous
release input retains precedence for historical-baseline checks.

The artifact manifest carries the pair's public publisher configuration and a content-addressed
reference to its private TLS/signing authority. Private authority is an assigned input object, never
result evidence. Producer admission verifies every release graph independently against the source,
host, backend and consumer formats. Worker object access and revocation cover the complete graph.
The clean consumer verifies authority hash/length, TLS identity/validity, and exact public configuration
before restoring the original loopback origin. Expired fixture authority requires rebuilding the pair.
Native helper state uses canonical paths, including macOS's `/tmp` to `/private/tmp` resolution.

Private update fixtures own a loopback HTTPS listener, temporary TLS material and an ephemeral
publisher. Their public configuration is compiled into acceptance packages before execution;
ordinary production packages cannot be relabelled as accepting that trust. Certificate trust is
limited to the launched test process, never installed in the host trust store. Publication copies
and verifies the exact candidate bytes before atomically offering a signed target/version. Failed
publication preserves the preceding offer. Signed request admission validates target metadata,
timestamp and bounded nonce replay state. Artifact capability routes expose only the owned copy,
support exact byte ranges and reject forwarded installation credentials. Scope release closes
the listener and removes its temporary trust and artifacts.
Corrupt-delivery scenarios first verify the admitted artifact, then alter only the fixture's copy
while preserving its length and authentic signed metadata. Recovery republishes the intact source;
neither malformed metadata nor a failed source-admission check substitutes for download rejection.
Settings persistence is checked across actual application process restarts using the same
isolated profile; a page reload cannot satisfy it. Each launch preserves its own trace and
process log, and the previous process is released before the next launches. Application quit
must traverse normal shutdown and terminate the main process with a successful exit code;
window hiding and forced cleanup cannot satisfy quit acceptance. Update restart
observes the exact retiring process and requires a clean exit; that observation alone cannot
qualify native replacement or automatic relaunch. Those require independent installed-byte and
replacement-owner evidence. Repeated service starts
retain the native application PID, service PID and service instance identity. After a
restart the previous owning service must have exited and the replacement must have
a new identity; tests never terminate a leftover process to manufacture acceptance.
CLI interruption acceptance first observes a connection to the deliberately stalled owned
service, then interrupts the CLI and requires the expected interruption exit. Early process
exit cannot satisfy this case. The service is resumed on every exit path, and afterward the
same application and service identities must remain usable. Unsupported native interruption
mechanisms block qualification rather than substituting forced termination.
Harness terminal paths use a real native PTY or ConPTY with an explicitly selected runtime
and isolated environment. Terminal interpretation preserves cursor movement, alternate screens,
Unicode and resize behavior; stripping escape sequences cannot substitute for rendered state.
Native child exit is distinct from transport exit, and forced cleanup cannot qualify normal exit
or user interruption. Sessions retain bounded transcripts and rendered evidence and reap owned
children on success, failure and cancellation. Terminal fixtures establish driver behavior only;
harness acceptance additionally requires its real model selection, generation and interruption.
The Pi terminal journey confirms keyboard selection, observes generated text absent from echoed
input, and requires a native persisted aborted assistant turn followed by a completed turn in
the same session with the selected provider/model. A turn that naturally finishes before Escape
fails interruption qualification. Session records and intermediate rendered screens are exported
alongside the terminal transcript; a standalone probe does not establish full worker qualification.
OpenCode similarly requires a still-active native assistant message before its two-key interrupt,
an exported abort error for that same message, and a distinct completed recovery message in the
same session. Presentation labels may guide the pinned third-party picker, but canonical native
provider/model IDs decide acceptance. Failed terminal journeys capture the rendered screen before
cleanup restores the alternate screen.
Hermes uses its native end-of-turn observer to distinguish keyboard interruption from completion,
with distinct turn IDs in the same session (the task ID may remain stable). The interrupted text
must render before Ctrl-C and be absent from echoed input. Hermes persists an interruption notice,
so its first assistant record has no successful finish reason; the second must finish with stop and
contain the rendered recovery answer. The exported billing endpoint must be the local application.
The observer and streaming display setting are scoped to the privately owned harness home and
restore its exact configuration after terminal cleanup. Missing or truncated native lifecycle
records fail acceptance; forced cleanup cannot satisfy normal exit.
Error scenarios preserve malformed harness files and restore the original fixture bytes
even after failed assertions. Visible error messages and file-specific repair guidance
may occupy separate elements. Service failure injection owns only an isolated loopback
listener; recovery requires successful startup after that listener is released.
Offline generation must establish an external connection before isolation and prove it fails
while a cached model is reloaded and generation is attested. Loopback remains available to the
application and its test client. Network isolation is allowed only for a qualified disposable
guest user and affects that user's external traffic, not the host management agent or other
users. An outward worker preserves only its trusted coordinator's resolved addresses and TCP
port, plus the configured DNS resolvers on port 53, so assignment revocation and cancellation
remain observable during the fault. These exceptions are recorded in isolation evidence; the
candidate cannot choose them. The independent model-host connection must still fail during
isolation. Loopback-only runners need no external exceptions. The fault owns its temporary rules; it never flushes shared firewall policy. Cleanup is
registered before installation, restoration is checked after success or failure, and cleanup
failures remain visible. A missing native isolation mechanism blocks the case.
Interrupted model acquisition must observe positive incomplete transfer progress before the
network cut and a rendered acquisition failure afterward. Recovery uses the UI retry, verifies
every target and companion file against the admitted catalog's revision, size and digest, and
attests generation while preserving app/service ownership. Removal and reacquisition are confined
to a qualified disposable guest; fault preflight and restoration precede model removal. An already
completed transfer cannot qualify interruption. Acquisition state and byte progress are semantic
automation attributes, independent of labels, styling and placement.
Resident-worker fault injection must verify the installed executable, inference-worker role and
ancestry under the owning service before terminating that exact process. Recovery observes the
failed model instance and explicitly reloads it; it never assumes automatic reload. A new native
worker generation must again match the admitted modules and requested backend while the app,
service and persistent inference-server identities remain unchanged. Before/fault/after evidence
is retained even when recovery fails. Unqualified native termination mechanisms block this case.
Screenshots are diagnostics rather than visual acceptance baselines. A presentation perturbation
probe verifies that copy and layout changes do not break functional paths.
Backend acceptance must correlate its public generation with the native completion and target-model
allocation evidence. Hardware enumeration, load-plan intent, or draft/projector GPU allocations
cannot qualify the target model's backend. Native completion diagnostics preserve the pre-aggregation
device locations; the lab must still validate runtime-module identity and request correlation.
The shared backend case combines these checks on one public generation and preserves the native
receipt even when allocation or module validation fails. CPU acceptance requires positive host
or explicitly identified CPU-device model allocation with no target-model accelerator allocation.
CPU-device allocations may coexist with accelerator allocations but cannot qualify GPU execution.
GPU acceptance requires positive
target-model allocation on the requested backend and hardware. Physical identifiers must match
host discovery exactly. Metal's absent physical identifier is accepted only for native index zero
on a host with exactly one enumerated Metal device; multiple-device ambiguity fails explicitly.
Requested layer counts are not a substitute for resident target-model allocation evidence.
The scoped execution collector accepts OTLP/JSON only on random loopback routes, retains only
reviewed ICN completion fields, bounds request and retained record sizes, and removes its listener
on scope exit. Exact exporter retries are idempotent; conflicting records invalidate collection.
Each observed generation has a fresh W3C trace propagated through the public ACN inference proxy.
The collector waits for that trace after generation succeeds without replaying generation, rejects
model mismatch and ambiguous native completions, and preserves both the public completion ID and
private native request ID rather than pretending they are the same identifier. Correlation alone
does not establish runtime-module identity or qualify the selected backend.
Loaded-module diagnostics retain filename, digest and length only when supplied. Absence remains
unknown; malformed supplied records invalidate collection. The native producer uses an already-loaded
full-path lookup, and the lab must independently compare its backing-file identity with the admitted
runtime. Expected module identities are computed from integrity-verified archives using the release
extractor. Missing, duplicate, unknown or byte-mismatched observed modules fail this comparison;
unused CPU variants need not be loaded. This comparison does not establish accelerator allocation.
Neither an on-disk module nor a same-named loaded module from another directory proves that
the selected installation executed.
JSON and JUnit reports derive from the same admitted plan and completed result.
JUnit distinguishes product failures from infrastructure errors; missing, blocked,
cancelled or duplicate selected results and cleanup failures cannot become green checks.
Finalized UI traces and command logs are published as content-addressed evidence before worker
removal. Custom selections retain shared desktop diagnostics even when they omit the install-launch
case. Evidence export failures remain visible without replacing the original case outcome.

Blocked prerequisites propagate without hiding the original product failure. Pass means every
required selected case passed and cleanup completed. Tests must demonstrate corrupt-input rejection,
source fidelity, cancellation, stale-worker fencing, allocation ambiguity recovery, backend mismatch
rejection and exact package consumption. Execution on one target never qualifies another.

Linux coverage is limited to Ubuntu, Debian, Fedora and Red Hat using existing DEB/RPM artifacts.
SUSE, Arch, Omarchy and Alpine packaging are future work. Provider credits and quotas do not confer
Windows licensing, signing identities or GPU availability; these are independently observed gates.
