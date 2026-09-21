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
Namespace input handoff stages the admitted graph locally with digest verification, then sends
one coordinator-created archive containing only digest-named regular objects and the invocation.
It does not issue a provider upload command per source file. Native extraction must succeed before
worker execution; extraction failures retain redacted diagnostics. The worker still verifies input
objects when consuming them. Candidate-produced archives are not accepted by this handoff.
Source candidates compile the release-owned CPU base and the selected host's backend pack alongside
the desktop. Metal and CUDA targets require their matching pack. Apple Silicon also includes
the Metal pack for CPU checks because ordinary application startup requires its distribution
metadata. Other CPU targets need only the base. This packaging requirement does not change
the requested execution backend: CPU checks still reject target-model GPU allocation. Compilation receipts bind the selected backend and native build identity. Final packaging
archives those exact compiled inputs and admits desktop, base and pack bytes together; it cannot
silently substitute a published inference runtime for unpublished source changes.
The manifest also includes the canonical archive of the bundled service, whose size is consumed
by ordinary runtime startup. Package admission validates that startup's bundle metadata is complete.
Windows source phases initialize the compiler through the admitted source's release-owned toolchain
helper and verify the matching Node import library before native dependency or application builds.
A preinstalled compiler alone does not establish the required build environment.
Build receipts bind the snapshot digest, base commit and native host; final manifests and every
package byte are verified before the installer can consume them. A base commit alone never
identifies dirty source. Source runs share one producer for each native artifact host and backend within that run.
A producer uses a canonical native CPU build host and a separately selected backend toolchain;
CUDA compilation does not implicitly allocate a GPU or use Spark. Build and test work have distinct
identities, fenced attempts, credentials and allocations even when they use the same target.
CUDA producers prepare an administrator-pinned compiler SDK in their owned build workspace.
Every component download is length- and digest-verified before extraction. Compiler identity,
required headers and a real kernel-to-PTX compilation must pass before candidate compilation;
this proves toolchain availability only, never driver readiness or GPU execution.
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

`verify` uses clean source, build output and consumer state. Warm `iterate` reuse and historical
`updateFrom` migration are not implemented; CLI and planning admission reject these requests
before uploading inputs or allocating workers. Source-built private update pairs remain supported.
No execution may touch a developer's normal application data. Local execution
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
Hermes file-tool qualification permits a failed call followed by a successful retry of that
tool in the same native turn. Every invocation needs a matching result, no failed tool may
remain unresolved, and the resulting file tree must match the exact intended change. Native
error events remain in the evidence. A completed tool turn without text deltas cannot qualify
streaming; streaming is asserted separately against an observed text-generating turn.
Terminal generation markers ignore letter case consistently in echo exclusion, rendered output
and persisted text. Native interruption, same-session recovery and successful completion remain
required; a marker or a completed first answer cannot establish cancellation.

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
Windows delivery runs the worker in the admitted local user's interactive desktop session.
A system-session process cannot qualify GUI behavior. A temporary, system-owned launcher
delivers only attempt-scoped authority through a protected input, validates the actual user and
session, observes the native task exit, and removes its task and credential files. Missing
interactive login, launch timeout or failed cleanup remains an infrastructure failure.
Fresh Azure Linux workers may prepare their trusted runtime through administrator-pinned
cloud-init configuration or a pinned Linux preparation recipe. A recipe names its OS distribution
and version: the allocator rejects a target mismatch before provisioning, and the guest rejects
an image mismatch before installing dependencies. The allocator verifies setup
bytes before creating resources and binds the preparation identity to the VM. A renewable recipe
pins archive identity and issues a fresh one-hour, blob-only read capability per allocation;
capability renewal does not alter the pinned preparation identity or grant provider credentials.
Native readiness requires completed cloud-init and the lab setup's final completion receipt.
Fatal cloud-init errors or a missing receipt fail admission. Recoverable platform warnings are
retained in detailed status and cannot substitute for the lab's successful completion receipt.
Namespace Mac preparation verifies the locked guest image and interactive user before installing
the trusted runtime and pinned harness dependencies. Preparation binds the admitted build/test
role into its receipt. Only a package producer must prove, before candidate delivery, that Finder
accepts automation used by the real installer layout. Clean package consumers do not request
Finder automation; native installation and app UI checks still run unchanged. On a qualified image with
existing accessibility authorization, it may approve only the management worker’s exact Finder
permission dialog through the normal UI. It cannot edit privacy databases, approve unrelated
prompts or skip packaging checks. The consent observer and probe have bounded lifetimes.
A root-owned receipt binds the preparation
recipe; reconciliation verifies that receipt and the live desktop without reinstalling tooling.
Incomplete preparation requires a fresh worker. Provider credentials stay on the coordinator;
only a bounded read capability for the pinned runtime enters the guest and is removed after use.
Native dependency failures retain diagnostic evidence before ordinary lease cleanup.
Windows client allocation requires an operator-verified licensing basis in administrator
configuration. Multitenant hosting emits Azure's Windows client license declaration; Visual Studio
dev/test eligibility does not imply that declaration. Credits, image visibility and Server diagnostic
success cannot establish client entitlement. Non-Windows targets reject client licensing fields.
Windows preparation separates pinned tooling, trusted runtime dependencies, and the admitted
user's desktop. Runtime readiness binds the runtime digest, native distribution/architecture and
user SID. One-shot login credentials are removed at logon before a desktop receipt is published;
that receipt also requires a live interactive session. Windows Server 2022 and 2025 are explicit baseline targets with license-included Azure
pricing and native Server identity checks. They cannot qualify Windows 10/11 client behavior. Azure records
recipe-bound preparation stages and observes ambiguous command submissions without repeating
installers. Restart intent is persisted before reboot so reconciliation cannot reboot an already
prepared desktop. Readiness is refreshed against the live session; an older command receipt cannot
satisfy a failed or ambiguous refresh. Stage execution and observation stay within the allocation
lease, and failure diagnostics are retained before cleanup.
NVIDIA guests require an explicit administrator-pinned driver recipe matching the target GPU and
Azure VM family. Unsupported OS/driver combinations fail before allocation. Driver downloads are
public Microsoft redistribution assets with length and digest checks. Windows installs drivers
before its one-shot desktop reboot and observes the live device afterward; reconciliation must
not repeat installation or reboot an already prepared desktop. Linux requires headers for its
running kernel. Native GPU model and driver version must match before candidate execution;
installer success alone cannot establish readiness or CUDA generation qualification.
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
Spark is opt-in for trusted source and exclusive within the lab. Its disposable Ubuntu ARM64
container uses the GB10 through NVIDIA CDI; reports identify the container distribution and
physical GPU separately. Source compilation stays on the Azure ARM64 producer. Docker over
SSH uses a pinned image, bounded CPU/memory/process limits and a lease lifetime. Transfers
and cleanup verify lease metadata plus the immutable container ID; cleanup never removes a
new container that happens to reuse the fixed lab name. The container grants native package
installation context only to its actual non-root user. It isolates and cleans up only
lab-owned files, processes and ports; it does not inspect or manage unrelated office workloads.
Network faults affect only the container's private network namespace. Update authorization uses
the real native policy service inside that container and only the packaged updater command.
Cloud access uses a coordinator-only Tailscale identity and authenticated SSH host discovery.
Enrollment credentials and network access never enter candidate containers. The connection
does not expose a public SSH port or require a privileged coordinator container. Enrollment
expiry is an operator-managed deployment prerequisite, independent of run credentials.

Installed-package ownership follows explicit present/absent transitions. Removal updates
ownership only after the native uninstaller succeeds; reinstall acquires new ownership.
Final cleanup removes only the currently owned installation. Dangling package launcher
symlinks count as removal failures even when their targets no longer exist.
Removal records the live application/service descendants with native process creation identity.
Verification checks those same instances after exit, including children orphaned by their parent;
PID reuse cannot count as a leak. Native inspection errors cannot count as absence. Login removal
first enables the installed user's real entry. Linux and Windows must then show an absent entry
or the unchanged command with its executable absent. macOS first records enabled native
SMAppService status from the packaged application, then requires that exact bundle and executable
to be absent; this establishes inability to launch, not deletion of Apple's opaque registration.
Reinstallation checks the saved login preference without silently re-enabling it.
Fixture replacement serializes removal and installation under the same ownership gate.
Failed removal retains the previous owner; failed replacement installation leaves no owned
installation and retains the requested package for an explicit subsequent attempt. Cancellation
cannot discard ownership of a native installation that completed during the transition.
Installing a replacement directly is fixture preparation, never proof of application self-update.
User-data retention compares the stopped application's complete isolated profile before and
after native removal, including file contents, directories and symbolic links without following
links outside the profile. Evidence contains hashes rather than file contents. Reinstall must
consume the same candidate and retain an application setting across the removal boundary. Linux login cleanup first enables the real installed-user entry and
captures the app/service descendant tree with process birth identities. Removal must leave no
captured process alive and either remove the login entry or leave the exact saved preference
dormant behind a missing TryExec executable. Reinstallation observes the retained enabled setting
without toggling it to manufacture success. These native assertions require an installed Linux
guest; other platforms must not report this qualification from an absent registration alone.

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
Debian development package trust verifies the admitted archive digest and length, compares every
installed payload file and symlink against native DEB extraction, and rehashes the admitted runtime
archives. A native archive listing must identify the standard unsigned container layout. Embedded
signatures require an independent expected-publisher policy and cannot be silently accepted as
unsigned; unknown members fail classification. Evidence explicitly records Unsigned and
productionTrusted=false. This does not establish APT repository trust, and production DEB trust
remains blocked until its publisher policy is configured.
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
verifies loader resolution and installation provenance. A pathname dependency cannot fall back to an
OS basename lookup when its owned file is missing. Fixture resolvers do not qualify native hosts.
System resolution admits explicitly allowed names only, requires a unique architecture-matching
loader-cache resolution, checks canonical OS-directory ownership and ELF architecture, and records
the installed DEB/RPM owner. An unpackaged x64 CUDA driver may instead carry a root-owned
receipt that binds its canonical path and current bytes to the exact driver payload extracted
from the administrator-pinned NVIDIA installer. The receipt must match the trusted runtime’s
installer version and digest; this exception applies only to `libcuda.so.1` on CUDA targets.
Other unowned system libraries fail. Multiple cache paths may qualify only when all resolve to the same
canonical OS file. Distinct cache files or hardware-capability choices block rather than
guessing loader precedence. Package ownership alone does not prove symbol-version compatibility.
Version requirements are matched by exact ABI identity against the resolved provider’s version
definitions, including owned dependencies. Missing or truncated version reports fail; a newer-looking
version number cannot substitute for an absent required identity.
Linux package inspection covers every installed native file and the admitted runtime composition.
Program headers must identify the architecture’s standard GNU loader when an interpreter is
present, and that loader must exist as an OS-directory ELF image of the correct architecture.
Only explicit desktop/system ABI names may cross the OS boundary; inference implementation and
CUDA toolkit libraries must remain owned. The NVIDIA driver library is admitted only for CUDA.
PE inspection uses the guest's API-set, side-by-side and architecture-aware native resolution.
Each owned root retains its loader context, including only the product's declared owned runtime
search directory, never a developer PATH. Ordinary and delayed imports must resolve through the
complete owned graph. OS boundaries require architecture-matching files in native Windows system
locations and a valid Microsoft signature; only CUDA may additionally admit NVIDIA's signed driver.
Candidate libraries are inspected, not executed, to establish this closure.
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
Updater cases share one scoped installation journey and restore the primary candidate before other
suites resume. The baseline records settings and model file hashes before replacement. The updater
must launch a new ready application/service owner at the admitted candidate version without a test
launch command; a new owner at the old version is recovery from a failed installation, not success.
Native package-database observation transfers cleanup ownership only within the same installation
paths and target. DEB payload checks compare extracted admitted files; RPM checks compare the
admitted package's SHA-256 file inventory. Both compare installed bytes and symlinks independently
of the updater's own hash check. Native RPM integrity checks do not establish publisher trust.
Post-update generation requires retained
settings and the original model hashes before loading again. Both release versions share one
scoped runtime origin so an automatically relaunched app can acquire its own admitted runtime.
An unresolved native handoff forbids further package mutation; disposable allocation cleanup remains
the final containment boundary.

Update download faults affect only the fixture's archive connections. They offer a proper prefix,
cut the response before its declared length, and reject further transfers until the fault scope
closes. Native-client tests observe received prefix bytes before triggering the cut. App acceptance
requires visible failure, the baseline still running, and successful intact download after recovery.
Fault restoration precedes subsequent cases even when assertions fail or the case is cancelled.

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

Linux desktop workers use the distribution's supported display server. A Wayland-only guest starts
a private compositor and requires an actual logical monitor before launching the worker. Its
session environment reaches the application and updater children; software composition of the
test display does not change the selected inference backend. Display processes and sockets are
released on failure and normal exit, with bounded failure diagnostics retained.

Linux coverage is limited to Ubuntu, Debian, Fedora and Red Hat using existing DEB/RPM artifacts.
SUSE, Arch, Omarchy and Alpine packaging are future work. Provider credits and quotas do not confer
Windows licensing, signing identities or GPU availability; these are independently observed gates.
