import { Clock, Context, Effect, ExecutionStrategy, Exit, Fiber, Option, Schema, Scope, Stream, SubscriptionRef } from "effect"
import { UpdateCandidate } from "@magnitudedev/release/hosted-update"
import type { DesktopUpdateState } from "@magnitudedev/client-common"
import { UpdatePreferences } from "./update-preferences"

export class ApplicationUpdateFailed extends Schema.TaggedError<ApplicationUpdateFailed>()("ApplicationUpdateFailed", {
  message: Schema.String,
}) {}
type Candidate = typeof UpdateCandidate.Type
export interface ApplicationUpdateSource {
  readonly check: Effect.Effect<Option.Option<Candidate>, ApplicationUpdateFailed>
  readonly download: (candidate: Candidate, progress: (completed: number) => Effect.Effect<void>) => Effect.Effect<string, ApplicationUpdateFailed, Scope.Scope>
  readonly stage: (archive: string, candidate: Candidate) => Effect.Effect<void, ApplicationUpdateFailed>
}
export const ApplicationUpdateSource = Context.GenericTag<ApplicationUpdateSource>("desktop/ApplicationUpdateSource")

const Transfer = Schema.Union(
  Schema.TaggedStruct("Idle", {}),
  Schema.TaggedStruct("Available", { candidate: UpdateCandidate }),
  Schema.TaggedStruct("Downloading", { candidate: UpdateCandidate, completed: Schema.Number, automatic: Schema.Boolean }),
  Schema.TaggedStruct("Cancelling", { candidate: UpdateCandidate }),
  Schema.TaggedStruct("Staging", { candidate: UpdateCandidate }),
  Schema.TaggedStruct("Ready", { version: Schema.String }),
  Schema.TaggedStruct("Failed", { message: Schema.String }),
  Schema.TaggedStruct("Closed", {}),
)
type Transfer = typeof Transfer.Type
type State = { readonly transfer: Transfer; readonly check: DesktopUpdateState["check"]; readonly preference: DesktopUpdateState["preference"] }
const present = (state: State): DesktopUpdateState => {
  const transfer = state.transfer
  switch (transfer._tag) {
    case "Available": return { ...state, transfer: { _tag: "Available", version: transfer.candidate.manifest.version, bytes: transfer.candidate.manifest.artifact.bytes } }
    case "Downloading": return { ...state, transfer: { _tag: "Downloading", version: transfer.candidate.manifest.version, completed: transfer.completed, total: transfer.candidate.manifest.artifact.bytes } }
    case "Cancelling": return { ...state, transfer: { _tag: "Cancelling" } }
    case "Staging": return { ...state, transfer: { _tag: "Staging", version: transfer.candidate.manifest.version } }
    default: return { ...state, transfer }
  }
}
export interface ApplicationUpdate {
  readonly state: Effect.Effect<DesktopUpdateState>
  readonly changes: Stream.Stream<DesktopUpdateState>
  readonly check: Effect.Effect<void, ApplicationUpdateFailed>
  readonly download: Effect.Effect<void, ApplicationUpdateFailed>
  readonly setAutoDownload: (enabled: boolean) => Effect.Effect<void, ApplicationUpdateFailed>
  readonly requireReady: Effect.Effect<void, ApplicationUpdateFailed>
  readonly close: Effect.Effect<void>
}
export const ApplicationUpdate = Context.GenericTag<ApplicationUpdate>("desktop/ApplicationUpdate")

/** Checks and transfers have independent admission; workers belong to the desktop owner. */
export const makeApplicationUpdate = (initialFailure: Option.Option<string> = Option.none()) => Effect.gen(function* () {
  const source = yield* ApplicationUpdateSource
  const preferences = yield* UpdatePreferences
  const preference = yield* preferences.read.pipe(Effect.match({
    onSuccess: autoDownload => ({ _tag: "Known", autoDownload }) as const,
    onFailure: error => ({ _tag: "Unavailable", message: error.message }) as const,
  }))
  const owner = yield* Scope.fork(yield* Scope.Scope, ExecutionStrategy.sequential)
  const state = yield* SubscriptionRef.make<State>({ preference, check: { _tag: "Idle" },
    transfer: Option.isSome(initialFailure) ? { _tag: "Failed", message: initialFailure.value } : { _tag: "Idle" } })
  const gate = yield* Effect.makeSemaphore(1)
  let transferWorker: Option.Option<Fiber.RuntimeFiber<void, never>> = Option.none()
  const closed = new ApplicationUpdateFailed({ message: "Magnitude is quitting." })
  const autoEnabled = (current: State) => current.preference._tag === "Known" && current.preference.autoDownload
  const failTransfer = (error: ApplicationUpdateFailed) => gate.withPermits(1)(SubscriptionRef.update(state, (current): State =>
    current.transfer._tag === "Downloading" || current.transfer._tag === "Staging"
      ? { ...current, transfer: { _tag: "Failed", message: error.message } } : current))

  // Caller owns the gate. Cancelling retains admission until scoped file cleanup has finished.
  const startDownload = (candidate: Candidate, automatic: boolean): Effect.Effect<void> => Effect.gen(function* () {
    yield* SubscriptionRef.update(state, (current): State => ({ ...current, transfer: { _tag: "Downloading", candidate, completed: 0, automatic } }))
    transferWorker = Option.some(yield* Effect.scoped(Effect.gen(function* () {
      const archive = yield* source.download(candidate, completed => SubscriptionRef.update(state, (current): State => current.transfer._tag === "Downloading"
        ? { ...current, transfer: { ...current.transfer, completed } } : current))
      const admitted = yield* gate.withPermits(1)(Effect.gen(function* () {
        const current = yield* SubscriptionRef.get(state)
        if (current.transfer._tag !== "Downloading") return false
        yield* SubscriptionRef.set(state, { ...current, transfer: { _tag: "Staging", candidate } })
        return true
      }))
      if (!admitted) return
      yield* source.stage(archive, candidate)
      yield* gate.withPermits(1)(SubscriptionRef.update(state, (current): State => current.transfer._tag === "Staging"
        ? { ...current, transfer: { _tag: "Ready", version: candidate.manifest.version } } : current))
    })).pipe(
      Effect.catchAll(failTransfer),
      Effect.catchAllDefect(() => failTransfer(new ApplicationUpdateFailed({ message: "The application update could not finish. Check again to retry." }))),
      Effect.ensuring(gate.withPermits(1)(Effect.gen(function* () {
        const current = yield* SubscriptionRef.get(state)
        if (current.transfer._tag !== "Cancelling") return
        yield* SubscriptionRef.set(state, { ...current, transfer: { _tag: "Available", candidate: current.transfer.candidate } })
        if (autoEnabled(current)) yield* startDownload(current.transfer.candidate, true)
      }))),
      Effect.interruptible, Effect.forkIn(owner),
    ))
  })
  const close = Effect.gen(function* () {
    yield* gate.withPermits(1)(SubscriptionRef.update(state, (current): State => ({ ...current, transfer: { _tag: "Closed" } })))
    // Finalizers can acquire the gate; never await them while holding it.
    yield* Scope.close(owner, Exit.void)
  }).pipe(Effect.uninterruptible)
  yield* Effect.addFinalizer(() => close)
  const check = gate.withPermits(1)(Effect.gen(function* () {
    const current = yield* SubscriptionRef.get(state)
    if (current.transfer._tag === "Closed") return yield* closed
    if (current.check._tag === "Checking") return
    yield* SubscriptionRef.set(state, { ...current, check: { _tag: "Checking" } })
    yield* source.check.pipe(
      Effect.flatMap(candidate => gate.withPermits(1)(Effect.gen(function* () {
        const current = yield* SubscriptionRef.get(state)
        if (current.transfer._tag === "Closed") return
        const replaceable = ["Idle", "Available", "Failed"].includes(current.transfer._tag)
        const next: State = { ...current, check: { _tag: "Succeeded", at: yield* Clock.currentTimeMillis },
          transfer: replaceable ? Option.isSome(candidate) ? { _tag: "Available", candidate: candidate.value } : { _tag: "Idle" } : current.transfer }
        yield* SubscriptionRef.set(state, next)
        if (replaceable && Option.isSome(candidate) && autoEnabled(next)) yield* startDownload(candidate.value, true)
      }))),
      Effect.catchAll(error => gate.withPermits(1)(SubscriptionRef.update(state, (current): State => current.transfer._tag === "Closed" ? current
        : { ...current, check: { _tag: "Failed", message: error.message } }))),
      Effect.catchAllDefect(() => gate.withPermits(1)(SubscriptionRef.update(state, (current): State => current.transfer._tag === "Closed" ? current
        : { ...current, check: { _tag: "Failed", message: "Could not check for updates." } }))),
      Effect.interruptible, Effect.forkIn(owner),
    )
  })).pipe(Effect.uninterruptible)
  return ApplicationUpdate.of({
    state: SubscriptionRef.get(state).pipe(Effect.map(present)), changes: state.changes.pipe(Stream.map(present)), close, check,
    requireReady: SubscriptionRef.get(state).pipe(Effect.flatMap(current => current.transfer._tag === "Ready" ? Effect.void
      : new ApplicationUpdateFailed({ message: "Download the application update before restarting." }))),
    download: gate.withPermits(1)(Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(state)
      if (current.transfer._tag !== "Available") return yield* new ApplicationUpdateFailed({ message: "Check for an available update before downloading." })
      yield* startDownload(current.transfer.candidate, false)
    })).pipe(Effect.uninterruptible),
    setAutoDownload: enabled => gate.withPermits(1)(Effect.gen(function* () {
      if ((yield* SubscriptionRef.get(state)).transfer._tag === "Closed") return yield* closed
      yield* preferences.write(enabled).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
      yield* SubscriptionRef.update(state, (current): State => ({ ...current, preference: { _tag: "Known", autoDownload: enabled } }))
      const current = yield* SubscriptionRef.get(state)
      if (enabled && current.transfer._tag === "Available") yield* startDownload(current.transfer.candidate, true)
      if (!enabled && current.transfer._tag === "Downloading" && current.transfer.automatic) {
        yield* SubscriptionRef.set(state, { ...current, transfer: { _tag: "Cancelling", candidate: current.transfer.candidate } })
        if (Option.isSome(transferWorker)) yield* Fiber.interruptFork(transferWorker.value)
      }
    })).pipe(Effect.uninterruptible),
  })
})

export const unavailableApplicationUpdate = (message: string): ApplicationUpdate => {
  const state: DesktopUpdateState = { transfer: { _tag: "Unavailable", message }, check: { _tag: "Idle" }, preference: { _tag: "Unavailable", message } }
  return { state: Effect.succeed(state), changes: Stream.succeed(state), check: new ApplicationUpdateFailed({ message }),
    download: new ApplicationUpdateFailed({ message }), setAutoDownload: () => new ApplicationUpdateFailed({ message }),
    requireReady: new ApplicationUpdateFailed({ message }), close: Effect.void }
}
