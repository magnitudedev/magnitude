---
applies_to:
  - packages/acn-protocol/src/boundary/onboarding.ts
  - packages/acn/src/onboarding/**
  - packages/acn/src/boundary/**
  - packages/client-common/src/onboarding/**
  - packages/client-common/src/local-models/setup*
  - packages/client-common/src/state/agent-client.ts
  - packages/client-common/src/state/client-services.ts
  - packages/client-common/src/hooks/use-onboarding-model-setup.ts
  - cli/src/app.tsx
  - cli/src/index.tsx
  - cli/src/features/model-setup/**
  - cli/src/commands/**
---

# Onboarding model setup

Onboarding model setup is a connection-scoped client flow composed over authoritative onboarding,
local-model, and model-slot services. It may satisfy first-run onboarding or be reopened voluntarily
after onboarding. Reopening is presentation intent and never reverses durable completion.

## Authority and ownership

ACN owns one durable, monotonic onboarding fact:

```text
Incomplete -> Complete
```

Completion is idempotent. It records that the first-run requirement was satisfied or dismissed; it
does not record whether a setup surface is open.

The onboarding, local-model, and model-slot client services each own their vertical Effect Query
domain. Their passive query Atoms are canonical server observations. Their Mutations perform RPC
commands and synchronize the acknowledged postcondition back into the canonical query. Query and
Mutation definitions remain private to the owning service.

The composed onboarding setup service owns only shared client interaction state and the lifetime of
the active cross-domain Effect. It defines no Query, Mutation, cache, mirror, or second copy of
server state. React and the CLI observe its one view and invoke semantic Effects; they do not own the
workflow or infer it from mutation history.

## Visibility and exit semantics

The surface is visible while onboarding is incomplete, explicit retained-open intent exists, or an
admitted setup operation has not terminalized. The exit action is derived from current durable
onboarding state:

- incomplete onboarding presents Skip and completes onboarding before closing;
- complete onboarding presents Close and clears retained-open intent without an onboarding RPC.

There is no stored Required/Requested origin. Repeated open is idempotent and cannot erase an active
operation or retained failure. Closed and Closing views do not depend on model or slot observations.
Initial onboarding uncertainty remains unavailable and is never interpreted as closed.

```text
Resting + incomplete ------------------------------> Open chooser (Skip)
Resting + complete + open -------------------------> Open chooser (Close)
Open chooser + select -> Prepare -> Install -> Assign -> Load -> [Complete onboarding] -> Closed
Open operation + cancel -> exact lower cleanup -------------------------------> Open chooser
Open operation + failure -------------------------------------------> Open chooser with notice
Open + Close --------------------------------------------------------> Closed
Open + Skip -> Complete onboarding ---------------------------------> Closed
```

## Operation and identity guarantees

Selection admission atomically publishes one nonterminal invocation and forks its terminalizing
worker into the setup service scope. The command acknowledges admission; it does not remain pending
for the worker's lifetime. The narrow admission commit is uninterruptible, while the forked worker
is explicitly restored to normal interruption so cancellation races and scope shutdown can work.

The worker passes exact predecessor outputs through one Effect program:

1. resolve the exact displayed serving configuration;
2. reconcile that configuration and retain the exact download admission when needed;
3. assign the exact returned provider model and captured reasoning effort to the primary slot;
4. invoke load for that captured selection;
5. accept readiness only for the same selection and configuration; and
6. complete onboarding only when it was incomplete at admission.

A stable snapshot of the submitted option is retained only as historical presentation evidence.
Canonical queries may replace it for display, but all decisions use exact identities and current
authoritative observations. Discovery refresh or removal therefore cannot make an active operation
unrenderable or redirect it.

Model-instance identity remains private to ACN's slot controller. Setup addresses the selected slot
and verifies its exact selection and configuration; it does not introduce a parallel instance API.
Selection replacement fails the invocation instead of satisfying or redirecting it.

Model-slot mutation scopes express independent command concurrency. Selection changes serialize
with selection changes, loads with loads, and stops with stops. Stop and selection replacement must
be able to reach ACN while a load request is pending because ACN owns cancellation and supersession.

## Cancellation, failure, and retry

Cancellation is cooperative and identity-safe. It cancels only the exact admitted download or asks
the authoritative slot owner to stop the captured slot load. The operation remains visible as
cancelling until cleanup and terminalization finish. Completing onboarding cannot be cancelled.

Every terminal update is fenced by invocation identity, so late work cannot overwrite a newer flow.
Expected lower failures remain typed and become one retained chooser notice. Defects are re-raised
only after client-owned nonterminal state is removed. The coordinator does not use `orDie` for
expected cleanup errors.

Query unavailability remains a failure of the public view, tagged with onboarding, local-model, or
model-slot ownership. Semantic retry invalidates only failed participating query domains. Query
failure is not converted into Closed or into an operation failure.

## Client integration

Reading, mounting, or remounting setup is observational. `/setup` sends the idempotent open event.
`--setup` is immutable launch configuration supplied when the connection-scoped client services are
built, so the setup service's initial retained state is already open before its first view can be
published. There is no post-render launch action, synthetic Result, or second setup state.

The CLI composition root resolves the shared setup-view Atom before the first React render. The
React hook and startup preflight use that exact client-keyed Atom, so the first rendered frame is
already Closed, Open, or Failure. The application gate still renders nothing if that view later has
no resolved value. Startup work is therefore never admitted from an unresolved onboarding
observation, completed onboarding never flashes setup, and requested setup never flashes the
ordinary app.

The public setup view has one top-level answer:

```text
Closed
Open {
  exitKind: Skip | Close
  content: Preparation | Chooser(operation?) | Closing
}
```

Repeated metadata lives on its parent. Choice options exist only on chooser content. The active
operation carries its model directly. The hook separately exposes hardware because hardware is an
orthogonal server observation, not setup lifecycle state.

## Conformance

- Opening never mutates durable onboarding.
- Successful required selection and Skip are the only setup paths that complete onboarding.
- Requested success and Close send no onboarding mutation.
- No nonterminal phase exists without a service-scoped terminalizing worker.
- Hook or renderer unmount does not interrupt an admitted operation.
- Service scope release interrupts owned work under normal Effect interruption semantics.
- Discovery refresh, failure, or removal cannot mask an active operation.
- Exact download admission governs observation and cancellation.
- Readiness must match the captured selection and configuration.
- Stop and selection replacement are not queued behind a pending load.
- Closed and Closing views do not observe unrelated model or slot queries.
- Query retry targets the failed participating owner.
- UI code renders one setup view and never inspects mutation history to derive lifecycle.
