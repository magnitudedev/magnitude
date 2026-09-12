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

The main destinations are Discover, Catalog, My Models, Connections, Status, and Settings. Service health and
model residency are distinct facts. There is no permanent bottom status strip. Discover, Catalog, My Models, and Connections occupy the top of the sidebar; Status and Settings stay pinned at the bottom. Discover leads with observed hardware, a Fast-to-Smart slider, and three distinct model recommendations with assessment-derived radar profiles. Each featured model uses its highest-ranked configuration; recommendations are full-width, vertically arranged, with room for their profiles and actions. Discover contains no catalog search or full catalog list. The separate Catalog destination owns the full collection, search, compatibility filtering, and expandable details with radar charts. My Models shows only acquisition/library entries, never the undownloaded catalog. Connections show the supported harnesses’ own artwork. Discovery uses the
curated ACN catalog, hardware observations, and the shared ranking algorithm; no renderer-owned
catalog, hardware inference, or independent recommendation source exists.
Settings reads the running application's version from the privileged host independently of service
readiness; it never substitutes a hardcoded version or the service protocol version.
Application update checks and downloads are explicit Settings actions. Main owns their work and
observation independently of the renderer and service readiness. Window Close and observer loss
cannot cancel admitted work. Mac updates use shared release-channel selection, verify the selected
archive's size and digest, and stage it through the native updater before reporting Ready. The local
staging endpoint exposes only that archive and closes on success, failure, or owner interruption.
Restart requires a visible window and a staged update at admission. Quit closes update admission and
cancels unfinished transfers before retiring the service; native installation/relaunch is invoked only
after owned-child cleanup and application-scope release. Ordinary Quit applies a staged update without
relaunch. Development profiles disable native update actions. Platform builds without an implemented
update transaction report that limitation rather than offering a nonfunctional restart action.
Before native staging, the owner durably records the selected target version and application bundle.
This is update intent, never a service lease or evidence of a live installer. On a later owner launch,
an older app exits before creating its service/window while that exact native installation is active.
The target version or a newer app admits normally and clears the receipt, including while its native
relaunch helper is still exiting. If installation has stopped without applying the target, the app
admits normally with a retryable update failure. Unknown native state cannot grant old-app startup.
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
restores a minimized window. Window close hides the retained renderer. Full Quit stops owned
children, proves cleanup, and releases application ownership. Cleanup failure stays visible
and retains ownership. The failure dialog offers Keep Open, Retry Quit, and an explicit Force Quit
that warns cleanup is unproven and exits unsuccessfully. Cancel never forces exit. Application control carries lifecycle intent and observation,
not another model API.
Fatal application initialization failure releases acquired resources, reports the original failure,
and exits unsuccessfully. Background failure cannot leave an inert process running without control;
foreground startup also presents the failure. Service-supervisor failures remain recoverable in
the running tray and window and do not use this fatal initialization path.

Connections observe actual provider configuration, required skills, and plugin integrity through the
privileged host. Saved connection receipts govern restoration and ownership, not the Connected label.
Filesystem access failures remain distinguishable from missing or overwritten configuration. Connect
writes configuration and required artifacts without launching a harness. Login startup is a separate
explicit preference. Headless commands and the desktop share one connector implementation.
Harness environment discovery is bounded asynchronous work, independent of tray and service startup.
It never mutates the application environment. Connections use the resolved search path, configuration
roots, and child-command environment; explicit command overrides remain authoritative. Failure falls
back to the inherited environment. Quit cancels the probe and retires its child process group.
Connect preserves the harness's
current model. A saved receipt exposes repair and removal even when configuration has been overwritten.
Intact configuration without a receipt can be Connected but is not claimed as removable owned state.
Connections remain observable while the inference service is unavailable: configuration integrity
does not depend on whether a model or service is running. Status shows separate service and model
sections. An unavailable model observation never means that no model is loaded; service failure
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
