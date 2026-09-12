---
applies_to:
  - packages/acn/src/service-lifecycle.ts
  - packages/acn/src/server.ts
  - packages/acn/src/server.test.ts
  - packages/acn/src/acn-subscriptions.ts
  - packages/acn/src/icn/**
  - cli/src/commands/server.ts
  - cli/src/commands/server-runtime.ts
  - cli/src/server/service.ts
  - packages/acn-protocol/src/schemas/acn-health.ts
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

The desktop application owns ACN as a direct child for the full application lifetime. ACN installs
native parent-loss protection and opens its inherited control channel before application or ICN
initialization. It reports Booted and waits for the owner to validate its retained child identity and
send Start. No SQLite owner row, competing candidate, adoption, or ownership polling participates in
normal serving. A missing owner channel fails startup.

The parent lifetime channel remains open after readiness. Owner loss terminates the owned service
tree; native protection does not depend on the JavaScript event loop. Only the desktop supervisor
may restart a failed service, after predecessor cleanup. Domain failures remain in their domains.

## Readiness and admission

Health, lifecycle observation, application RPC dispatch, and shutdown read
one lifecycle value. The control server exists throughout startup. Application RPC rejects until
the complete application and private ICN exist.

`Ready` installs the RPC application atomically. `Stopping` closes RPC dispatch before becoming
observable. The first stop reason wins; both transitions are
monotonic and idempotent.

The desktop bounds startup and recovery. ACN independently owns a five-minute absolute
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

RPCs, subscriptions, sessions, requests, and observation do not determine process presence. Closing
an ordinary client cannot stop the service. Closing the desktop window hides it; full application
Quit is the owner shutdown command. ICN remains the private mandatory child.

## Shutdown

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
