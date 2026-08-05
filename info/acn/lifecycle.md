# ACN lifecycle

The ACN is Magnitude's local daemon. Clients use SDK JIT ensurance to obtain one compatible,
canonical ACN for the Magnitude data root. There is no separate coordinator process: local clients
perform filesystem and process mechanics directly, while desktop main and the web host perform the
same mechanics for their renderer or browser.

The durable source of truth is in:

- [`design/acn/lifecycle/jit-spawning.md`](../../design/acn/lifecycle/jit-spawning.md)
- [`design/acn/lifecycle/client-lifecycle.md`](../../design/acn/lifecycle/client-lifecycle.md)
- [`design/acn/lifecycle/service-lifecycle.md`](../../design/acn/lifecycle/service-lifecycle.md)

## JIT ensurance

`ensure(target)` keeps reconciling until a compatible ACN is `Ready`, unless the machine cannot
make progress because of a typed terminal condition such as an unavailable artifact or an exact
process tree that cannot be reaped.

Clients coordinate concurrent launches through process-safe machine state. Only the current fenced
spawn claim may publish or replace an ACN. A new candidate is never spawned until the preceding ACN
and its ICN are proven exited.

## Client upgrades

Every client begins with its bundled ACN target. When it observes a compatible higher-priority ACN
in `Starting` or `Ready`, it immediately adopts that target for the rest of its process lifetime.
Recovery and spawning use the adopted target, so an older open client cannot revive its older ACN.

## Timing

The principal timings are intentionally unrelated:

| Timing | What it detects | What expiry means |
| --- | --- | --- |
| 2-second publication grace | Another client may be between launch-related filesystem/process work and observable early ACN registration. | Contend for the fenced spawn claim and revalidate; it is not proof of ACN failure. |
| 30-second startup stall window | One exact `Starting` ACN has made no authoritative phase or measured-progress advance. | Fence and reconcile that attempt, then remove it exactly before replacement. |
| 5-minute absolute startup ceiling | ACN application startup has not completed even if its reported phases keep changing. | ACN commits startup failure and begins its own terminalization. |

The publication grace ends as soon as `Starting` or `Ready` is visible. The stall window resets only
on real phase or monotonic measured progress. The five-minute ceiling never resets. Application RPC
duration is not a lifecycle deadline.

## Service lifecycle

One ACN owns one monotonic service lifecycle:

```text
Starting(activity, progress?) -> Ready -> Stopping(reason) -> exact exit
```

`Ready` atomically opens application admission. `Stopping` atomically closes it. ACN shutdown closes
sessions and subscriptions, terminates and reaps ICN, and retains machine ownership until its owned
runtime is gone. If graceful cleanup stalls, the client holding the exact replacement claim
escalates through terminate and kill; timeout never counts as proof of death.

## Recovery

A concrete transport failure causes the client to re-observe ACN state. It does not by itself prove
that the selected ACN died. A compatible successor is selected, a compatible `Starting` successor
is observed within its bounded window, and true absence enters JIT ensurance.

Queries, observations, and mutations retain their own retry semantics. In particular, a mutation
whose response was lost is not blindly replayed unless its domain contract makes that safe.
