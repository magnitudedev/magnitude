---
applies_to:
  - inference/crates/icn-api/**
  - inference/crates/icn-server/**
  - packages/icn-protocol/**
  - packages/sdk/src/acn-jit/**
  - packages/acn/src/service-lifecycle.ts
  - packages/acn/src/server.ts
  - cli/src/**
  - desktop/src/**
  - web/src/**
---

# Client-independent model residency

ACN process lifetime, ICN child lifetime, client lifetime, session-runtime lifetime, and model
residency are independent concerns. Clients have no heartbeat, lease, connected count, or presence
state. Closing a client releases only client-owned selection and transport resources; it never
mutates ACN, ICN, or model lifetime.

ACN remains alive until explicit administrative stop, replacement, ownership loss, signal,
startup failure, mandatory ICN loss, or fatal process failure. ICN remains ACN's private mandatory
child and has no independent idle shutdown.

ICN owns physical model residency. Every accepted inference request owns an exact lease on the
Ready instance until completion, failure, or cancellation. A Ready instance has one actor-owned
monotonic idle deadline exactly when it has zero inference leases. Production uses a fixed
one-hour interval:

- becoming Ready with zero leases starts a full interval;
- acquiring a lease clears the deadline;
- releasing the final lease starts a fresh full interval;
- equivalent explicit warm-load demand starts a fresh full interval; and
- expiration gracefully releases the instance only if it is still Ready with zero leases.

The residency actor serializes inference admission and deadline expiration. It owns one deadline,
not sleeping timer tasks or generation fences. Client presence, HTTP connections, ACN RPCs,
observations, downloads, session residency, loading time, and unrelated work do not refresh it.
There is no mutable residency-policy API.

Explicit Stop, replacement, memory-pressure eviction, worker failure, and ICN teardown remain
distinct release causes. Session runtimes keep their own two-minute zero-use plus Quiescent
offloading policy. Subscription keepalives remain transport-liveness frames, and admitted shared
operations such as downloads outlive initiating callers according to their domain ownership.

## Guarantees

- First-party and external inference requests protect residency through the same exact lease.
- A Ready instance with an active lease cannot idle-release.
- Final lease release and equivalent warm demand each start a full one-hour interval.
- Closing any or all clients cannot stop ACN or ICN and cannot change a model deadline.
- Unrelated activity cannot retain or prematurely release a model.
- Idle expiration cannot affect a replacement instance.
