---
applies_to:
  - cli/src/index.ts
  - cli/src/commands/**
  - cli/src/server/**
  - cli/src/runtime/**
  - cli/src/startup/**
  - cli/src/update/**
  - packages/sdk/src/client.ts
  - packages/daemon-management/src/desktop-native/application-client.ts
  - packages/daemon-management/src/desktop-native/application-host.ts
---

# Headless CLI startup

The CLI is a finite, noninteractive command surface. Bare invocation prints help. Help, version,
documentation, connection inspection, and service status are observational and never start the
application. There is no terminal renderer, onboarding preflight, update prompt, or agent harness.

## Application ownership

Commands validate argument syntax and supported identifiers before requesting startup.
Service-backed commands ask the installed desktop application to run in the background and await
its exact Ready service and compatible RPC version. The same request applies to cold and warm
startup. It never shows, restores, or focuses a window. Only explicit `magnitude app open` sends
ShowWindow; this does not wait for inference readiness and can open a failed application's Status.

The privileged application client owns installation discovery, launch intent, and local control.
Only an absent application control endpoint permits a launch attempt. A timeout, permission error,
unresponsive owner, or explicit application failure is not absence and must not create another
owner. Concurrent launches coalesce through the desktop native lifetime lock. A cancelled CLI
request does not cancel an application that has already started. Cold startup observes launcher
failure until application control responds. Nonzero launcher exit fails promptly; the Linux
installation guard reports package-manager repair guidance. A successful platform dispatcher may
exit before application admission and is not treated as failure. This short-lived observation
never supervises or terminates the desktop process.

The CLI does not install or download ACN, elect a daemon owner, kill a predecessor, or register an
independent OS daemon. Missing installation and missing graphical session produce actionable errors.
Windows cold launch checks the native assigned desktop; a noninteractive caller may control an
existing desktop owner but cannot create an invisible owner in its own session.
Development uses its isolated desktop, service endpoint, data, and harness configuration.
An explicitly isolated profile also applies to packaged CLI runs: application control, service
requests, and harness configuration use that profile together. Choosing a private profile does
not change whether startup launches a source checkout or an installed application.

## Service administration

`service start` ensures the desktop in the background, awaits compatible Ready, acknowledges, and
exits. It does not enable login startup. `service stop` asks the application to Quit and awaits
that exact process occurrence's exit. The desktop proves owned-child cleanup; the CLI never stops
ACN independently. An already absent application is a successful stop.

`service status` observes application lifecycle without starting it. Model observations are separate
from service readiness, and unavailable model evidence must not be presented as no loaded model.
Login installation registers the desktop's graphical-session startup; uninstallation unregisters
it and requests full application shutdown while preserving model files and settings.

## Recovery and updates

An established SDK connection may reconnect to an available service, but cannot invoke its starter
again. A stale request or subscription therefore cannot undo explicit Quit. A fresh command may
explicitly ensure the application again. Startup waits remain bound to the admitted application
occurrence and fail if it is replaced.

CLI update commands delegate to the desktop update owner independently of service readiness.
Status is passive; active commands may start the tray without opening the window. The CLI does not
run a package manager or update itself separately. RPC mismatch gives an update action rather than
replacing or downgrading the running service.
Before a cold installed macOS launch, the host waits for an active native update job targeting that
exact application bundle to finish. An inactive retained job is not an active installation. Observation
failure or a bounded wait expiring fails the command without starting the old application. The CLI
does not stop, replace, or supervise the installer, and warm control requests retain their existing
application semantics.

## Acceptance

- Every retained command terminates without terminal UI or prompts.
- Passive commands neither create a desktop process nor alter login registration.
- Background cold and concurrent launches preserve window visibility and focus.
- Service readiness uses the exact application's compatible service, independent of model loading.
- Cancelling startup leaves an already admitted desktop alive.
- Quit stops the application and its owned tree; established clients cannot resurrect it.
- Installation and login startup never register a standalone daemon.
