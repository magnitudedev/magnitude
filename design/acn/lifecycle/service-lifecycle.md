---
applies_to:
  - packages/acn/src/service-lifecycle.ts
  - packages/acn/src/server.ts
  - packages/acn/src/server.test.ts
  - packages/acn/src/acn-subscriptions.ts
  - packages/acn/src/icn/**
  - cli/src/commands/status.ts
  - cli/src/commands/status-runtime.ts
  - cli/src/server/application.ts
  - packages/acn-protocol/src/schemas/acn-health.ts
  - packages/acn/src/remote-access.ts
  - packages/acn/src/remote-access.test.ts
  - packages/acn-protocol/src/schemas/remote-access.ts
  - packages/sdk/src/remote-access.ts
---

# ACN service lifecycle

One ACN process owns one authoritative service lifecycle:

```text
Starting(activity, progress?) -> Ready -> Stopping(reason) -> exact exit
```

`Exited` is observed externally through exact process identity. `Installing` is a client
presentation of `Starting`, not an admission mode. Startup or process-authority failure enters
`Stopping`; a bounded application resource such as one session runtime cannot determine process
lifetime.

## Process admission

The admitted Desktop or Headless application owns ACN as a direct child for its full lifetime. ACN installs
native parent-loss protection and opens its inherited control channel before application or ICN
initialization. It reports Booted and waits for the owner to validate its retained child identity and
send Start. No SQLite owner row, competing candidate, adoption, or ownership polling participates in
normal serving. A missing owner channel fails startup.

The parent lifetime channel remains open after readiness. Owner loss terminates the owned service
tree; native protection does not depend on the JavaScript event loop. Only the application supervisor
may restart a failed service, after predecessor cleanup. Domain failures remain in their domains.

## Readiness and admission

Health, lifecycle observation, application RPC dispatch, and shutdown read
one lifecycle value. The control server exists throughout startup. Application RPC rejects until
the complete application and private ICN exist.

`Ready` installs the RPC application atomically. `Stopping` closes RPC dispatch before becoming
observable. The first stop reason wins; both transitions are
monotonic and idempotent.

The application owner bounds startup and recovery. ACN independently owns a five-minute absolute
application-startup ceiling; optional progress cannot extend it. Expiry enters Stopping(startup-failed).

## Per-user application

Login launches the desktop process in the graphical user session with background intent. The app
owns the tray and service; its optional visible window has no effect on that ownership. OS startup
registration does not install a second independently supervised ACN. Explicit Quit stops the owned
tree and exits the desktop; login preference remains for the next login without immediate restart.

ACN binds one public loopback endpoint, normally 127.0.0.1:10100. Isolated development/test owners
may supply another explicit port. Health is available during startup. RPC dispatch is admitted only
at Ready and fenced by the selected instance ID. Inference paths remain unchanged. The inherited
control channel carries startup health and shutdown; there is no discoverable coordination listener.

ACN serves the browser app at `/`, even while it starts, so a browser can show that state. Release
executables embed the app's files; a service run from source serves the `web` build directory and
otherwise answers `/` with build instructions. Hashed assets are cached indefinitely and `index.html`
never; a path without a file extension is an app page and receives `index.html`; paths under the
service's own routes are never answered with the app; nothing outside the app's files is served.
HTML carries a same-origin Content-Security-Policy. The plain-text summary of the inference API is
served at `/inference`.

Network access is off by default and read once from `network` in `config.json` when ACN starts.
When enabled, ACN also listens on all interfaces, or on one configured address beside loopback, so
local clients are never displaced. Loopback is decided by socket peer address, never by header, and
loopback callers never sign in. A remote caller gains application control only by signing in from a
browser: `POST /auth/session` with the Network access key, compared in constant time, sets an
`HttpOnly`, `SameSite=Strict` session cookie (`Secure` over HTTPS), and `GET /auth/session` reports
whether the caller is local, signed in, or must sign in, and whether a key exists to sign in with.
`/rpc` admits loopback callers and signed-in callers whose `Origin` matches their `Host`; anonymous
remote callers get 401. `/health` returns the instance identity only to loopback and signed-in
callers, so the SDK's admission works unchanged in a signed-in browser; anyone else learns readiness
only. Sessions live in ACN memory, expire after 30 days unused, and end with `POST /auth/session/end`
or when ACN restarts. Because every network change, including a regenerated key or disabling
network access, applies only when ACN restarts, that restart is also what signs every browser out.
Failed sign-ins are throttled per remote address: after five, each attempt waits twice as long as
the last, up to one minute; a success clears the count. With no key configured, nobody can sign in.
Inference routes require the generated API key as a Bearer token or `x-api-key` from remote callers
unless the key requirement is switched off, and never from loopback callers; the session cookie
never authorizes inference, and `requireApiKey` never affects sign-in. The Host header is accepted only for
local names, and with network access on also for IP literals, `host.docker.internal`, `.ts.net`
names, and names listed under `network.allowedHosts`; no wildcard is ever accepted. CORS stays
loopback-only. Harness connections keep writing the loopback origin.

RPCs, subscriptions, sessions, requests, and observation do not determine process presence. Closing
an ordinary client cannot stop the service. Closing the desktop window hides it; full application
Quit is the owner shutdown command. ICN remains the private mandatory child.

## Shutdown

The application owner presents the user-facing shutdown reason. Routine administrative ACN shutdown
is a debug diagnostic; unexpected shutdown reasons remain visible at the normal logging level.

Every stop cause uses one process-owned, single-flight shutdown:

```text
commit Stopping and close admission
  -> terminate subscriptions and transports
  -> close application and session scopes
  -> terminate and reap private ICN
  -> exit ACN
  -> desktop owner proves its child tree absent before restart or full Quit
```

`beginStopping` completes immediately after the atomic stopping/admission transition; it never
awaits drain, finalizers, child shutdown, or exit. The server-owned supervisor performs every later
step with a fixed deadline and disconnects cooperative work before applying its timeout, so an
uninterruptible finalizer cannot retain escalation. Abrupt ACN loss closes ICN's private parent pipe;
ICN is not durably recorded or reconciled by the external manager. The ACN never removes or
reassigns its own ACN occurrence.

The inherited control channel accepts Shutdown before application readiness. Native parent-loss
protection remains armed throughout startup, serving, and teardown. Every public health value and
control-channel Health observation derives from the same authoritative lifecycle.
The ordered health forwarder belongs to the enclosing owner lifetime, not the fallible application
scope. Before closing application scope, ACN gives that forwarder a bounded two-second
window to finish reporting Stopping and receive the desktop acknowledgement. Startup failure cannot cancel that final report merely because
the application acquisition failed first. Channel failure or the deadline still permits teardown;
this delivery boundary neither delays the atomic stopping transition nor proves process exit.

## Guarantees

- One lifecycle governs health, readiness, RPC dispatch, and shutdown.
- No application or ICN work precedes Start from the retained parent channel.
- Parent ownership remains mandatory after admission; no admitted detachment exists.
- No steady-state owner database, election, or owner polling remains in ACN.
- A bounded domain failure cannot retire the process or interrupt unrelated domains.
- Window close and ordinary client close preserve serving; full owner Quit stops the tree.
- Cooperative teardown and exact owned-process escalation are independently bounded.
