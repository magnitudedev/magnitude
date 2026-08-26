---
applies_to:
  - packages/acn-protocol/src/boundary/onboarding.ts
  - packages/acn/src/onboarding/**
  - packages/acn/src/boundary/**
  - packages/acn/src/model-commands.ts
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

The composed onboarding setup service owns only shared client interaction state, including the local
model ranking preference, and the lifetime of the active cross-domain Effect. It defines no Query,
Mutation, cache, mirror, or second copy of server state. React and the CLI observe its one view and
invoke semantic Effects; they do not own the workflow or infer it from mutation history.

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
Open chooser + select -> Prepare -> Install -> Assign -> Ensure resident -> Select harness
Select harness + Magnitude -> Apply options -> [Complete onboarding] -> Closed
Select harness + external -> Apply options -> Connect harness -> [Complete onboarding] -> Handoff
Open operation + cancel -> detach observation --------------------------------> Open chooser
Open load + expected instance failure ------------------------------> Retained failed load
Open operation + unexpected failure -------------------------------> Open chooser with notice
Open + Close --------------------------------------------------------> Closed
Open + Skip -> Complete onboarding ---------------------------------> Closed
```

## Operation and identity guarantees

Selection admission atomically publishes one nonterminal invocation and forks its terminalizing
worker into the setup service scope. The command acknowledges admission; it does not remain pending
for the worker's lifetime. The narrow admission commit is uninterruptible, while the forked worker
is explicitly restored to normal interruption so cancellation races and scope shutdown can work.

The worker passes exact predecessor outputs through one Effect program:

1. retain the exact displayed canonical model ID;
2. install that Model and retain the exact native Download admission when needed;
3. assign that exact model ID and captured reasoning effort to the primary slot;
4. admit native residency for that model ID through the shared ICN coordinator;
5. accept readiness only for the exact returned Instance and model ID; and
6. retain that exact model while the user selects a destination;
7. configure the selected destination through `HarnessConnection`; and
8. complete onboarding only after all selected destination work succeeds.

A stable snapshot of the submitted option is retained only as historical presentation evidence.
Canonical queries may replace it for display, but all decisions use exact identities and current
authoritative observations. Discovery refresh or removal therefore cannot make an active operation
unrenderable or redirect it.

Instance identity is native ICN state. Setup uses the returned identity only to observe the admitted
occurrence; it never persists it in the Slot. Slot selection and native residency remain independent
authorities and are joined by canonical model ID for presentation.

## Cancellation, failure, and retry

Cancellation is cooperative and identity-safe. Before installation admission it interrupts the
request; after a Download is admitted it cancels only that exact Download where shared-work rules
permit. After residency admission it detaches this setup waiter and does not stop the shared
Instance. Completing onboarding cannot be cancelled.

Every terminal update is fenced by invocation identity, so late work cannot overwrite a newer flow.
An expected model-instance failure remains typed and terminalizes as a retained failed-load result
containing the exact attempted model. Retry starts a new invocation for that model; Choose another
returns to the unlocked chooser without changing the installed-model inventory. In particular,
low-memory is an actionable runtime outcome rather than an unexpected setup error.

ICN command failures remain on the Effect error channel. ACN maps generated-client remote and
transport errors into the typed mutation failure without stringifying the wrapper object. When a
load command fails, setup briefly reconciles the canonical slot lifecycle so an authoritative typed
instance failure wins over the less-specific command failure. A replaced selection is rejected
immediately, and bounded reconciliation with no instance outcome retains the command failure.
Defects are re-raised only after client-owned nonterminal state is removed. The coordinator does not
use `orDie` for expected cleanup errors.

Other terminal setup failures retain both the typed failure and its semantic subject. Model-operation
subjects include the exact attempted model and operation; setup-wide subjects cover exit and harness
work. This retained result lets presentation produce a useful unexpected-error sentence rather than
showing an RPC or generated-client wrapper tag.

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
  content: Preparation | Chooser(operation?) | Harness | ApplyingHarness | HarnessHandoff | Closing
}
```

Repeated metadata lives on its parent. Choice options and the connection-scoped Fast-to-Smart
preference exist only on chooser content. Its normalized value is clamped to `[0, 1]`, survives
renderer remounts for the connection, and are not durable onboarding state. The active
operation carries its model directly. Harness content carries the captured ready model and one
ordered detection snapshot. The hook separately exposes hardware because hardware is an
orthogonal server observation, not setup lifecycle state.

The CLI renders every Open state as the entire application viewport. Chat, its timeline, composer,
activity rail, file panel, and overlays are not rendered behind setup. Every stage occupies the same
centered, borderless setup frame so its origin and available bounds do not move as the workflow
advances. The persistent progress indicator sits directly above the stage content and contains
`Choose model`, `Install model`, and `Select harness`; upcoming markers are empty, active markers are
filled in accent blue, and completed markers/connectors are white. It is horizontal while the model
chooser uses its side-by-side layout and becomes vertical at the chooser's existing stacked-layout
breakpoint.

Stage content does not repeat the active step as a second title. In particular, the model chooser
begins with its hardware context beneath the progress indicator rather than rendering a redundant
`Choose a local model` heading.

The chooser places the non-focusable Fast-to-Smart scale after hardware and before model rows. Its
fixed three-row region keeps the interaction hint beside the track in the wide layout and moves it
onto the third row only in the stacked layout. It shows at most ten ranked eligible models regardless
of installation state, followed by every installed model under `ON THIS COMPUTER`. An installed
model may therefore appear in both groups. Keyboard
traversal visits only model rows; Left/Right and `h`/`l` adjust Fast-to-Smart regardless of which
model row is selected. Re-ranking preserves the cursor's row/rank position rather than following the
previous model identity. A ranked row's installation state
determines whether its action is Loaded, Load, or Download.
Ranking input and model input are locked during an active operation. Hardware failure does not
fabricate a memory maximum or selectable ranked results.

Each clipped model section overlays dim overflow affordances within its fixed model-row viewport;
it reserves no indicator rows. The lower affordance replaces the last otherwise-visible model row
while models remain below. After scrolling, the upper affordance replaces the first otherwise-visible
model row and reports the exact number above. An absent affordance leaves neither a blank line nor
extra height, so the first model always sits immediately below its section heading at the top.

The shared frame height is the model chooser's required row count, not a percentage of the terminal.
Its wide and stacked variants account for their respective progress and chooser layouts, and the
removed duplicate model title contributes no reserved row. Every later stage reuses that computed
height. Hardware context contributes its actual wrapped row count at the current frame width. Fixed
title, progress, ranking-control, operation, and footer regions never shrink into overlapping rows.

The frame reserves one shared footer position across stages. Stage guidance and interaction hints
use the same two-line typography there, so changing stages does not move or restyle the footer. One
empty row is guaranteed immediately before the footer without changing the frame's total height.
The detail pane reserves rows only for the selected model title, one-line summary, radar, and an
applicable memory warning; it never reserves operation rows. Starting an operation replaces the
Fast-to-Smart control region with a fixed three-row operation region while keeping the selected
model details and radar visible. The radar animates whenever its derived model values change,
including changes caused by re-ranking at the retained cursor position. The frame therefore has
exactly the same height while choosing,
downloading, and loading. The first operation row contains the status and any useful transfer
measurements, the second contains the progress bar, and the third contains controls. Loading and
downloading both expose `Cancel (Esc)` and replace that control with an inline Yes/No confirmation.
A retained failed load keeps the same three rows: the first names the model and concrete failure,
the second keeps the progress bar but colors it red, and the third offers Retry loading and Choose
another model. Low-memory text states the additional memory the user must free. Escape chooses
another model; Left/Right select an available recovery action and Enter performs it. Non-retryable
failures omit Retry. Stage-explanation subtext is not rendered. Loading begins at zero percent until
authoritative fractional progress advances it. The active operation label uses an eased shimmer for
starting, downloading, cancelling, configuring, preparing, stopping, loading, verifying, and
finishing states. Its cosine highlight scales with the text length, crosses muted text in about
850 ms, and remains absent for about 950 ms between sweeps; it never changes text weight. Failure
text remains static. Unexpected setup errors render in failure color beneath the shared control
hints, wrapping to the setup body width. Those rows increase the frame only while the error exists;
normal setup height reserves no error space. The message naturally names the attempted model and
operation when that context exists and includes the underlying typed failure detail.
The harness stage is one unboxed linear menu containing supported destinations followed by the
startup and skill toggles. Harness rows are ordinary single-choice rows, not checkbox controls.
Up and Down traverse every enabled control in that order; unavailable destinations remain visible
but cannot receive focus. Space changes a focused toggle. Enter from any row continues with the
currently selected harness and the current toggle values; no separate Continue row is rendered.

An external destination produces `HarnessHandoff` only after selected startup and skill work,
adapter reconciliation, and durable onboarding completion succeed. The CLI runtime observes this
state outside React, gracefully closes the client lease, unwinds the root and renderer scope, and
only then starts the returned executable with inherited terminal I/O and the captured model active.

## Conformance

- Opening never mutates durable onboarding.
- Successful required destination application and Skip are the only setup paths that complete onboarding.
- Requested success and Close send no onboarding mutation.
- No nonterminal phase exists without a service-scoped terminalizing worker.
- Hook or renderer unmount does not interrupt an admitted operation.
- Service scope release interrupts owned work under normal Effect interruption semantics.
- Discovery refresh, failure, or removal cannot mask an active operation.
- Exact download admission governs observation and cancellation.
- Readiness must match the captured selection and configuration.
- Expected instance failures remain typed, retain the exact failed load, and expose recovery in place.
- Unexpected notices preserve their typed failure and semantic operation/model subject.
- Cancellation after residency admission never stops shared residency work.
- Closed and Closing views do not observe unrelated model or slot queries.
- Query retry targets the failed participating owner.
- UI code renders one setup view and never inspects mutation history to derive lifecycle.
- Ranking controls are connection-scoped setup state and never persisted or sent to ACN.
- Downloadable ranking is derived only from authoritative hardware, model facts, and the chooser controls.
- A ready model advances to harness selection; model readiness alone never completes onboarding.
- External harness launch never overlaps the Magnitude renderer.
