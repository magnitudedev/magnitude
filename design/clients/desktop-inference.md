---
applies_to:
  - desktop/src/*.ts
  - desktop/src/renderer*
  - desktop/src/*.tsx
  - desktop/src/desktop-rpc.ts
  - desktop/src/electron-rpc.ts
  - desktop/src/login-startup.ts
  - packages/sdk/src/desktop-host.ts
  - packages/client-common/src/desktop/**
  - packages/harness-connections/**
---

# Desktop inference application

The existing Electron application is the local inference product. Its retained renderer owns one
SDK connection and one client-common Effect Query runtime. Electron main owns the tray and service
supervisor; ACN owns the inference engine. Renderer recreation rereads ACN and never restarts it.
Every platform exposes full Quit in its native application menu, including when no tray host exists.
Window Close remains hide-only; menu Quit uses the same awaited owner shutdown as tray and CLI.
Unix SIGTERM requests that same awaited shutdown; it does not bypass child retirement.
macOS/Linux system-shutdown notification requests cleanup without vetoing OS termination or opening
a shutdown-error dialog. The OS may allow less time than normal Quit; native lifetime containment
remains the fallback. Windows confirmed session end can terminate Electron synchronously, so an
asynchronous event handler is not a graceful-shutdown guarantee; owned jobs must contain that exit.
Main permits three automatic renderer-crash retries. After exhaustion or a failed main-document load,
model observations become unavailable and the tray retains Open and Quit. Explicit Open retries
the renderer with a renewed budget. Background demand never renews it or shows a window. Renderer
failure cannot terminate or restart the service, and teardown never starts renderer recovery.
The privileged IPC transport identifies each preload/runtime occurrence independently of its retained
window. Crash or main-document navigation disconnects that occurrence and releases its subscriptions.
Messages carry the renderer-session identity; retired requests and queued replies cannot cross into
another occurrence. Same-document navigation preserves the current session. Only the main frame
may access the host RPC transport. Finite host calls declare replay policy; state-changing host
actions are at-most-once and lost replies are not replayed into a replacement renderer session.

The main destinations are Discover, Catalog, My Models, Connections, Usage, Status, and Settings. Service health and
model residency are distinct facts. There is no permanent bottom status strip. Discover, Catalog, My Models, Connections, and Usage occupy the top of the sidebar; Status and Settings stay pinned at the bottom. Discover leads with observed hardware, a Fast-to-Smart slider with five visible stops and clickable labels, and five recommendations with assessment-derived radar profiles, allowing up to two configurations per model in score order. Discover presents one shared panel with five compact numbered selection rows beside a single selected-model profile. Only the selected model shows actions; Profile and Details controls switch the right pane without adding a full-width disclosure footer. The best match is selected initially and when the ranking preference changes. Selection is local presentation state and never loads a model. If a selected recommendation disappears, the first available recommendation supplies the profile. Radar charts place each metric value beside its axis label, without a separate values list. Discover contains no catalog search or full catalog list. The separate Catalog destination owns the full collection, search, compatibility filtering, and expandable details with radar charts. My Models shows only acquisition/library entries, never the undownloaded catalog. Connections show the supported harnesses’ own artwork. Discovery uses the
curated ACN catalog, hardware observations, and the shared ranking algorithm; no renderer-owned
catalog, hardware inference, or independent recommendation source exists.
Settings reads the running application's version from the privileged host independently of service
readiness; it never substitutes a hardcoded version or the service protocol version.
Main owns scheduled Magnitude-hosted update checks, automatic downloads and explicit Settings actions
independently of the renderer and service readiness. The automatic-download preference does not
disable checks. Window Close and observer loss cannot cancel admitted work. Magnitude-owned user data
lives under the shared `.magnitude` root: canonical config owns the automatic-download preference,
root identity.pem owns request identity, electron/ owns Electron userData and sessionData configured
before profile initialization, state/ owns application coordination, and updates/ owns one installer
and one update.json. Isolated development/test roots preserve the same layout.

A complete verified download publishes the exact public release and an installation union:
Unattempted, Attempted, or Failed with a bounded reason. No persisted state claims readiness,
installer liveness or success. Ordinary Quit cancels incomplete downloads and retains prepared
updates. macOS does not stage Squirrel until installation is requested. The local staging endpoint
exposes only that archive and closes on success, failure, or interruption.

Before ACN starts, startup reconciles native exclusion and the installed version. Reaching or passing
the saved release retires both files regardless of the saved outcome. Only Unattempted may install
automatically; a background Linux launch defers interactive authorization without marking an attempt.
Startup and explicit restart/retry use the same offline signature/size/hash verification and durable
Attempted write before native invocation. Explicit restart first retires owned children while retaining
application ownership. A failed attempt retains bytes and its actual reason; an unresolved Attempted
record displays “The update did not complete” and requires explicit retry. Retry never redownloads
valid retained bytes. Desktop settings and `magnitude update discard` can remove a retained download
under the same owner admission; failed cleanup preserves the previous presentation so it can be
retried. Discard does not change update preferences. Missing or invalid evidence cannot authorize installation.

Platform helpers hold a native installation lease across owner exit, installer execution and
same-release outcome recording, releasing it before relaunch. Competing bootstrap exits without
waiting while holding application ownership. An older Mac app defers while its native installer is
active; an installed target or newer version may reconcile successfully during native relaunch.
Unknown native state cannot grant old-app startup. Foreground intent belongs to the live handoff,
not another persisted update field. The Mac helper captures native job identity before staging,
waits for Squirrel to finish replacement, and launches the app with the retiring environment and
explicit window intent. Native automatic relaunch must not discard an isolated profile or show a
window after background startup. Cleanup is recoverable and cannot race an active installer.
Development profiles disable native update actions unless explicitly built for isolated acceptance.
Linux retains the signed package for an explicit handoff to the system package manager. The user
helper acknowledges readiness before owner exit, then waits on the inherited lifetime channel.
An explicit update action may request authorization even while the window is hidden; background
startup defers that prompt. Relaunch window intent is independent of authorization permission.
Polkit authorizes only the privileged package operation. That operation verifies installed,
root-owned publisher trust, target/version, copied package bytes and native package identity before
invoking the package manager. Existing installation admission excludes another running app owner;
an authorization or package-manager failure never means success. The helper relaunches as the
original user, preserving whether the window was open, and records a retryable failure when needed.
It is transient installation work, not an independent service owner or a login service.
The packaged Linux desktop adopts the launcher's shared installation descriptor before starting
children and marks it close-on-exec. The desktop retains package admission until process exit;
service and update-helper children cannot inherit the lease and block their own installation.
Recommendations order fitting assessed configurations using the shared preference. The remaining
curated catalog stays discoverable with explicit pending, incompatible, or insufficient-memory
explanations. Details expose catalog license/source links, capabilities, context, and labeled
performance estimates. Source links open HTTPS destinations in the system browser. Loading a
different model asks explicitly before replacing observed active residency.

The desktop session service owns navigation and bridges native menu actions into the same model
mutations used by the window. Window and tray model commands pass through the shared local-model
service; hooks expose actions and command status without exposing mutation atoms to the renderer.
Tray model text is a disposable projection of the canonical model
query. Loss of renderer observation disables model actions; it cannot imply no model is loaded.
Main independently displays owned-service startup, recovery, failure, and shutdown.
Stop remains available for an observed Stopping model, allowing explicit escalation or retry after
cleanup failure. Window actions suppress duplicate requests while their mutation is pending and
display declared model failure messages without internal Effect stacks; unknown failures use
actionable fallback wording. ACN resolves both active and slot Stop against Stopping as well as Loading
and Ready instances, so retry never silently succeeds without contacting the retained instance.
Main retains the current tray menu independently of its native icon. Linux host recovery replaces
the preceding icon before creating one replacement and restores the latest menu. Host loss never
stops serving, destroys the window, or requests activation. Menu updates and replacement are serialized;
application shutdown terminalizes tray ownership so late observations cannot recreate an icon.
Status and application control expose tray registration separately from service/model state. Registered
means native registration was requested successfully, not guaranteed pixel visibility or user pinning.
The tray shows service and model state without a setup-completion state. First use follows the ordinary Discover, model and connection actions.

A background launch starts the owner and tray without showing or focusing a window. Explicit Show
Window or navigation intent opens it. Dock activation reopens a hidden window and explicit Open
restores a minimized window. Raising or reactivating the retained window preserves its selected page;
only an explicit destination from a native menu action requests navigation. Window close hides the retained renderer. Full Quit stops owned
children, proves cleanup, and releases application ownership. Cleanup failure stays visible
and retains ownership. The failure dialog offers Keep Open, Retry Quit, and an explicit Force Quit
that warns cleanup is unproven and exits unsuccessfully. Cancel never forces exit. Application control carries lifecycle intent and observation,
not another model API.
Fatal application initialization failure releases acquired resources, reports the original failure,
and exits unsuccessfully. Background failure cannot leave an inert process running without control;
foreground startup also presents the failure. Service-supervisor failures remain recoverable in
the running tray and window and do not use this fatal initialization path.

Connections observe actual provider configuration, required skills, and plugin integrity through the
privileged host. Detected installations appear first, with installation and Magnitude connection status shown separately. Within installed harnesses, connected entries appear first with a green dot and their configuration paths; disconnected entries use a neutral dot and do not present expected paths as existing configuration. Unverifiable configuration remains an explicit unknown state. Undetected harnesses appear below with their artwork, Not installed status, and an official installation link; configuration details and connection actions are hidden until installation is detected. Detection refreshes automatically and has no manual Detect action. Saved connection receipts govern restoration and ownership, not the Connected label.
Filesystem access failures remain distinguishable from missing or overwritten configuration. Connect
writes configuration and required artifacts without launching a harness. Login startup is a separate
explicit preference. Headless commands and the desktop share one connector implementation.
Harness environment discovery is bounded asynchronous work, independent of tray and service startup.
It never mutates the application environment. Connections use the resolved search path, configuration
roots, and child-command environment; explicit command overrides remain authoritative. Failure falls
back to the inherited environment. Quit cancels the probe and retires its child process group.
Connect preserves the harness's
current model. For an installed harness, Connect repairs configuration that has been overwritten; Disconnect removes an intact managed connection. Connection status sits beneath the harness title, with connection actions in the header. Only connected cards have a divided configuration-details section.
Intact configuration without a receipt can be Connected but is not claimed as removable owned state.
Connections remain observable while the inference service is unavailable: configuration integrity
does not depend on whether a model or service is running. Status combines service readiness, model residency, and active downloads in one top section while keeping their states distinct. Idle downloads add no empty section. An unavailable model observation never means that no model is loaded; service failure
also suppresses stale model controls. Model stopping uses the same canonical mutation as the tray
and model library.
Status displays observed model-loading progress and current download, update, and removal activity.
Unknown progress stays indeterminate; service or query unavailability cannot become an idle claim.

Connection observations refresh after mutations and periodically while observed, so edits made by
other tools become visible. Unrelated user fields do not invalidate a connection. Development profiles
isolate harness configuration and skills from the user's normal profile.

## Visual identity

The inference rewrite preserves the existing desktop/web visual identity. Reuse the existing
appearance initializer and `magnitude.appearance` preference, with System, Light, and Dark choices.
The initializer installs the canonical client-common palette variables; importing Tailwind alone
does not initialize that palette. Native window appearance follows the same selected preference.

Use the existing slate surfaces and blue actions/selections, Inter UI text, Martian Mono headings,
shared Magnitude mark, existing Lucide/Phosphor icons, and shared button/input/progress primitives.
The Magnitude mark is a transparent white outline in dark appearance and its black inverse in
light appearance. Tray artwork uses the transparent outline; macOS template rendering follows the
menu-bar appearance. Model identities use the landing site's family/provider artwork throughout
Discover, My Models, active-model status and download activity. Action and navigation icons retain
their existing semantics. Missing provider artwork must not be replaced with another company's logo.
Do not create a separate palette, substitute fonts, or copy a second appearance store. Desktop
sources must be included in Tailwind scanning. Both light and dark treatments require live visual
acceptance, including disabled controls, focus, progress, failures, and native window chrome.
Discover keeps its hardware and preference controls separate from model assessment. Only its combined
recommendation list and profile panel uses an assessment skeleton, with the live settled/total count
inside the panel. Partial rankings are withheld until all assessments settle; the completed panel
fades in without an external status row shifting its position. Hardware discovery uses photo, name, and specification placeholders matching the
shared hardware card layout while unobserved. Reduced-motion preferences disable the fade.
Every destination retains its page shell during initial loading. Other pending observations use skeletons
with the same card geometry and responsive breakpoints as their content; independent sections settle
independently. Available content stays visible during refresh, and failures replace skeletons with
explicit unavailable states. Placeholder values never imply zero usage, no models, or a disconnected
harness. Loading regions announce once, expose no actions, and respect reduced motion. Unknown
collection lengths and installation states remain unknown until observed.

## Login startup

Settings observes OS registration rather than a saved boolean. Failed host actions preserve their
actionable message across the preload boundary without displaying internal Effect stacks. macOS registers the main application
through SMAppService and detects login launch before deciding window visibility. Windows registers
the installed executable with `--background` and observes startup approval for its stable application
identity. Command matching and approval must refer to that same per-user entry; a different entry
for the executable cannot substitute for it. Executable paths containing spaces remain intact. Linux uses one user XDG
autostart entry, honors desktop exclusions and disablement, and writes a Hidden override when disabled
so a lower-priority system entry cannot re-enable startup. No login adapter requests automatic OS
restart after Quit. Native platform acceptance includes signed macOS login and Windows startup approval;
filesystem simulation does not establish those guarantees.

Status uses a green check for a ready service and keeps model residency separate. Hardware recommendation artwork belongs to Discover, not Status. Normal background behavior is explained in user terms; native tray registration terminology stays out of the healthy-state UI. Development/test login-startup restrictions explicitly identify the build as such.

Status presents a single Memory section using authoritative active-model residency. The headline is the sum of model weights, KV cache, and overhead across the allocation's memory domains; overhead consists of compute and auxiliary allocations. No loaded model means zero in all three categories and the headline. Loading or unavailable residency is shown explicitly, never as an estimated allocation. OS process footprint and whole-machine usage are not part of this display. The section has no measurement disclosure or explanatory footer.

Discover supplements inference hardware with a read-only native description of the client device. Enclosure identity selects bundled manufacturer product photographs through explicit manufacturer/model matches; it never supplies ranking, memory capacity, or processor capabilities. Unknown or placeholder identity cannot imply a known enclosure. A graphics-card image represents its hardware family and cannot identify a laptop GPU or complete PC. Product shots show one clearly visible device on an empty background, never a color lineup or multiple-angle composite, with no visible credits, license labels or reference captions. Source provenance remains in the asset inventory. Photographs remain available offline and do not assert an unobserved finish or internal configuration. Shared enclosures reuse the same image across processor variants.

The Usage destination exposes persistent [local serving usage](../acn/local-serving-usage.md): input, cached input and output tokens, Today/All time, model filtering, measured generation speed and first-token latency. Status contains operational state and memory, not usage statistics. Incomplete or unavailable evidence is explicit. Usage recording belongs to the service and continues while the window is closed.
