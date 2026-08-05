---
applies_to:
  - packages/sdk/src/acn-jit/**
  - packages/acn/src/daemon-registration.ts
  - packages/acn/src/machine-ownership.ts
  - packages/acn/src/peer-acn.ts
  - packages/acn/src/process-identity.ts
  - packages/acn/src/daemon-lifecycle.ts
  - packages/acn-protocol/src/acn-registry.ts
  - packages/acn-protocol/src/schemas/acn-health.ts
  - desktop/src/main.ts
  - web/scripts/dev-server.ts
---

# JIT ACN ensurance and upgrades

For one Magnitude data root, clients converge on one canonical ACN. There is no separate
coordinator process: the SDK performs JIT ensurance locally, while desktop main and the web host
perform the same operation for clients without filesystem and process access.

`ensure(target)` means that, while the host remains capable of running the target, the operation
keeps reconciling until a compatible ACN is `Ready`. Missing publication, a hung `Starting` ACN, an
abandoned launch, and another client disappearing are intermediate conditions, not successful
absence or reasons to give up. An unreapable process, unavailable artifact, or failed host primitive
is a typed terminal failure because automatic convergence is no longer possible.

## Targets and client upgrade

Compatibility and precedence are separate:

| Identity | Meaning |
| --- | --- |
| Coordination protocol | Whether clients and ACNs understand the machine-state envelope |
| RPC protocol | Whether a client may use the ACN |
| Storage protocol | Whether the ACN may use the data root |
| Release target | Reproducible artifact and deterministic replacement priority |

Product version is diagnostic metadata unless it identifies a reproducible release target.
Development targets use stable artifact identity rather than wall-clock build time.

Each client starts with its bundled target and owns a monotonic `effectiveTarget`. Observing a
compatible higher-priority ACN in either `Starting` or `Ready` immediately advances that target.
All later selection, recovery, artifact resolution, and spawning use `effectiveTarget`; it never
moves backward during the client process lifetime.

Consequently, an older client may continue against a compatible newer ACN, but can never revive its
older ACN afterward. If the newer startup fails, that client ensures another instance of the newer
effective target. Incompatible clients fail locally rather than launching a competing ACN.

## Machine coordination

The shared data root contains only the state needed for clients and ACNs to coordinate:

```text
spawn claim               exact client process, token, target, lease
canonical registration    exact ACN identity, target, endpoint, process identity
machine owner             exact ACN runtime owner
instance records          exact processes available for reconciliation and cleanup
```

The spawn claim serializes mutation across independent clients. Acquisition revalidates canonical
state. A live claim cannot be bypassed merely because a waiter is impatient. If the claim owner
stops renewing its bounded lease, takeover publishes a new fencing token; the former owner can no
longer publish, replace, or remove anything.

An ACN may publish canonical state only for the current claim and target. Publication is
process-safe and rejects a lower-priority target when an equal or higher target is canonical or
being admitted. Scheduling and last-writer order cannot move the target backward.

Registration, health, RPC dispatch, shutdown, and cleanup bind the exact ACN and process-start
identity. A PID or URL alone never grants authority. Missing, malformed, unreadable, or timed-out
evidence is reconciled under the spawn claim; it is not silently converted into authoritative
absence.

## Ensurance

Every initial connection and recovery uses the same operation:

```text
Ready compatible ACN
  -> advance effectiveTarget if needed
  -> return it

Starting compatible ACN
  -> advance effectiveTarget if needed
  -> observe that exact startup within its bounded window

No usable ACN
  -> allow the publication grace
  -> acquire the spawn claim and revalidate
  -> join any resulting acceptable startup or spawn effectiveTarget

Starting attempt exceeds its window
  -> acquire/take over the spawn claim and revalidate
  -> fence, stop, and reap that exact attempt
  -> spawn effectiveTarget
```

An expired deadline authorizes revalidation and takeover, never an uncoordinated second spawn. A
new candidate is not started until the previous candidate and its owned ICN are proven exited. If
exact removal cannot be proven, ensurance fails as blocked rather than accumulating processes.

The desired outcome is a compatible canonical endpoint, not survival of a particular child. If
another client wins with an acceptable target, every waiter succeeds with that ACN. Caller
cancellation ends only that observation; it does not make an already spawned process unowned.

## Timing semantics

Every duration answers one question and has one expiry action:

| Timing | Justification | Expiry permits |
| --- | --- | --- |
| 2-second publication grace | Covers the short local gap in which another client has begun launch-related filesystem/process work but its early ACN registration is not yet observable. It is coalescing grace for file publication and process scheduling, not an ACN health judgment. | Contend for the spawn claim and revalidate. It does not prove that an ACN died or permit bypassing another live claim. |
| 30-second startup stall window | Bounds trust in an exact, observable `Starting` ACN. Thirty seconds is deliberately much larger than normal local phase transitions while detecting a candidate hung in backend preparation far sooner than the process-wide ceiling. | Take over ensurance, revalidate the exact candidate, and begin its bounded removal. It does not permit a concurrent spawn. |
| 5-minute absolute ACN startup ceiling | Bounds the entire ACN-owned application startup even if phases or reported progress continue changing. It prevents progress churn or a defective client from retaining `Starting` forever. | The ACN commits `Stopping(startup-failed)` and terminalizes its process-owned runtime. External ensurance then reaps or replaces it. |

The publication grace begins only while no usable registration is visible and ends immediately when
an exact `Starting` or `Ready` ACN appears. It is not restarted by repeated empty reads.

The startup stall window is scoped to one exact ACN instance. A real phase transition or monotonic
measured progress restarts it; repeated health responses, estimated animation progress, or unchanged
values do not. A replacement instance receives its own window. The five-minute ceiling never
restarts.

Transport request duration is not one of these clocks. A slow application RPC does not imply ACN
failure and cannot initiate JIT replacement.

## Exact replacement and cleanup

Replacement first fences the exact canonical attempt so it can no longer win publication or admit
new work, then performs:

```text
request cooperative shutdown
  -> bounded exact-exit wait
  -> revalidate and terminate
  -> bounded exact-exit wait
  -> revalidate, force kill, and reap/prove death
  -> prove the owned ICN exited
  -> remove only exact instance records
  -> permit the next spawn
```

Timeout advances this escalation or returns a typed unreapable-process failure. It never proves
death. Delayed cleanup validates identity again and cannot remove a successor.

## Guarantees and verification

- JIT ensurance converges to `Ready` or a typed condition that makes convergence impossible.
- A compatible higher target immediately and permanently upgrades every observing client's
  effective target.
- A lower target cannot displace or follow an admitted higher target.
- Publication grace, startup stall, absolute startup, and cleanup bounds are never interchangeable.
- At most one client owns spawn mutation and at most one ACN owns the runtime.
- No replacement spawns before exact ACN and ICN cleanup completes.
- A hung client, launch claim, ACN startup, or cleanup path cannot block ensurance forever.
- Observation uncertainty and application latency do not independently authorize mutation.

Conformance covers independent CLI, desktop, and web clients; older clients recovering while a
newer ACN remains `Starting` beyond two seconds; every claim and startup owner hanging; unchanged
and advancing progress; failure at the five-minute ceiling; caller exit; PID reuse; and cleanup that
cannot prove ICN exit. Tests assert the invariants after every transition, not only at completion.
