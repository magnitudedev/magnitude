import { Context, Effect, ExecutionStrategy, Exit, Option, Schema, Scope, Stream, SubscriptionRef } from "effect"
import { FSM } from "@magnitudedev/utils"
import { DesktopUpdateCandidate } from "@magnitudedev/release"
import type { DesktopUpdateState } from "@magnitudedev/client-common"

export class ApplicationUpdateFailed extends Schema.TaggedError<ApplicationUpdateFailed>()("ApplicationUpdateFailed", {
  message: Schema.String,
}) {}
type Candidate = typeof DesktopUpdateCandidate.Type
export interface ApplicationUpdateSource {
  readonly check: Effect.Effect<Option.Option<Candidate>, ApplicationUpdateFailed>
  readonly download: (candidate: Candidate, progress: (completed: number) => Effect.Effect<void>) => Effect.Effect<string, ApplicationUpdateFailed, Scope.Scope>
  readonly stage: (archive: string, candidate: Candidate) => Effect.Effect<void, ApplicationUpdateFailed>
}
export const ApplicationUpdateSource = Context.GenericTag<ApplicationUpdateSource>("desktop/ApplicationUpdateSource")

class Idle extends Schema.TaggedClass<Idle>()("Idle", {}) {}
class Checking extends Schema.TaggedClass<Checking>()("Checking", {}) {}
class Current extends Schema.TaggedClass<Current>()("Current", {}) {}
class Available extends Schema.TaggedClass<Available>()("Available", { candidate: DesktopUpdateCandidate }) {}
class Downloading extends Schema.TaggedClass<Downloading>()("Downloading", { candidate: DesktopUpdateCandidate, completed: Schema.Number }) {}
class Staging extends Schema.TaggedClass<Staging>()("Staging", { candidate: DesktopUpdateCandidate }) {}
class Ready extends Schema.TaggedClass<Ready>()("Ready", { version: Schema.String }) {}
class Failed extends Schema.TaggedClass<Failed>()("Failed", { message: Schema.String }) {}
class Closed extends Schema.TaggedClass<Closed>()("Closed", {}) {}
type State = Idle | Checking | Current | Available | Downloading | Staging | Ready | Failed | Closed
const machine = FSM.defineFSM({ Idle, Checking, Current, Available, Downloading, Staging, Ready, Failed, Closed }, {
  Idle: ["Checking", "Closed"], Checking: ["Current", "Available", "Failed", "Closed"],
  Current: ["Checking", "Closed"], Available: ["Checking", "Downloading", "Closed"],
  Downloading: ["Staging", "Failed", "Closed"], Staging: ["Ready", "Failed", "Closed"],
  Ready: ["Closed"], Failed: ["Checking", "Closed"], Closed: [],
} as const)
const present = (state: State): DesktopUpdateState => {
  switch (state._tag) {
    case "Available": return { _tag: "Available", version: state.candidate.version, bytes: state.candidate.artifact.bytes }
    case "Downloading": return { _tag: "Downloading", version: state.candidate.version, completed: state.completed, total: state.candidate.artifact.bytes }
    case "Staging": return { _tag: "Staging", version: state.candidate.version }
    default: return state
  }
}
export interface ApplicationUpdate {
  readonly state: Effect.Effect<DesktopUpdateState>
  readonly changes: Stream.Stream<DesktopUpdateState>
  readonly check: Effect.Effect<void, ApplicationUpdateFailed>
  readonly download: Effect.Effect<void, ApplicationUpdateFailed>
  readonly requireReady: Effect.Effect<void, ApplicationUpdateFailed>
  readonly close: Effect.Effect<void>
}
export const ApplicationUpdate = Context.GenericTag<ApplicationUpdate>("desktop/ApplicationUpdate")

/** Admission is finite. Its worker belongs to the application, never the requesting renderer. */
export const makeApplicationUpdate = (initialFailure: Option.Option<string> = Option.none()) => Effect.gen(function* () {
  const source = yield* ApplicationUpdateSource
  const owner = yield* Scope.fork(yield* Scope.Scope, ExecutionStrategy.sequential)
  const state = yield* SubscriptionRef.make<State>(Option.isSome(initialFailure) ? new Failed({ message: initialFailure.value }) : new Idle({}))
  const gate = yield* Effect.makeSemaphore(1)
  const failed = (error: ApplicationUpdateFailed) => SubscriptionRef.update(state, current =>
    current._tag === "Checking" || current._tag === "Downloading" || current._tag === "Staging"
      ? machine.transition(current, "Failed", { message: error.message }) : current)
  const work = (effect: Effect.Effect<void, ApplicationUpdateFailed>) => effect.pipe(
    Effect.catchAll(failed),
    Effect.catchAllDefect(cause => Effect.logError(cause).pipe(Effect.zipRight(failed(new ApplicationUpdateFailed({ message: "The application update could not finish. Try checking again." }))))),
    Effect.interruptible, Effect.forkIn(owner), Effect.asVoid,
  )
  const close = gate.withPermits(1)(Effect.gen(function* () {
    yield* SubscriptionRef.update(state, current => current._tag === "Closed" ? current : machine.transition(current, "Closed", {}))
    yield* Scope.close(owner, Exit.void)
  })).pipe(Effect.uninterruptible)
  yield* Effect.addFinalizer(() => close)
  return ApplicationUpdate.of({
    state: SubscriptionRef.get(state).pipe(Effect.map(present)), changes: state.changes.pipe(Stream.map(present)), close,
    requireReady: SubscriptionRef.get(state).pipe(Effect.flatMap(current => current._tag === "Ready" ? Effect.void
      : new ApplicationUpdateFailed({ message: "Download the application update before restarting." }))),
    check: gate.withPermits(1)(Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(state)
      if (current._tag !== "Idle" && current._tag !== "Current" && current._tag !== "Available" && current._tag !== "Failed") {
        return yield* new ApplicationUpdateFailed({ message: "An update is already in progress or Magnitude is quitting." })
      }
      yield* SubscriptionRef.set(state, machine.transition(current, "Checking", {}))
      yield* work(source.check.pipe(Effect.flatMap(candidate => SubscriptionRef.update(state, current => current._tag !== "Checking" ? current
        : Option.isSome(candidate) ? machine.transition(current, "Available", { candidate: candidate.value }) : machine.transition(current, "Current", {})))))
    })).pipe(Effect.uninterruptible),
    download: gate.withPermits(1)(Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(state)
      if (current._tag !== "Available") return yield* new ApplicationUpdateFailed({ message: "Check for an available update before downloading." })
      const candidate = current.candidate
      yield* SubscriptionRef.set(state, machine.transition(current, "Downloading", { candidate, completed: 0 }))
      yield* work(Effect.scoped(Effect.gen(function* () {
        const archive = yield* source.download(candidate, completed => SubscriptionRef.update(state, current => current._tag === "Downloading"
          ? machine.hold(current, { completed }) : current))
        yield* SubscriptionRef.update(state, current => current._tag === "Downloading" ? machine.transition(current, "Staging", { candidate }) : current)
        yield* source.stage(archive, candidate)
        yield* SubscriptionRef.update(state, current => current._tag === "Staging" ? machine.transition(current, "Ready", { version: candidate.version }) : current)
      })))
    })).pipe(Effect.uninterruptible),
  })
})

export const unavailableApplicationUpdate = (message: string): ApplicationUpdate => ({
  state: Effect.succeed({ _tag: "Unavailable", message }), changes: Stream.succeed({ _tag: "Unavailable", message }),
  check: new ApplicationUpdateFailed({ message }), download: new ApplicationUpdateFailed({ message }),
  requireReady: new ApplicationUpdateFailed({ message }), close: Effect.void,
})
