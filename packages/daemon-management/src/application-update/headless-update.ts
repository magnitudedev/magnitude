import { Deferred, Effect, Option, Stream } from "effect"
import type { UpdateCheckReason } from "@magnitudedev/release/hosted-update"
import { ApplicationUpdateControlFailed, type ApplicationUpdateAction, type ApplicationUpdateState } from "@magnitudedev/sdk/desktop-host"
import type { ApplicationUpdate } from "./application-update"
import { makeUpdateSchedule } from "./update-schedule"

/**
 * What an install request does while `magnitude serve` owns updates. A CLI-only installation installs
 * at the server's next idle point (`WhenIdle`); any other installation explains how it is updated.
 */
export type HeadlessInstall = { readonly _tag: "WhenIdle" } | { readonly _tag: "Refuse"; readonly message: string }

export const packageInstallRefusal: HeadlessInstall = { _tag: "Refuse",
  message: "To install the update now, stop `magnitude serve` and run `magnitude update install`." }

const preparing = (state: ApplicationUpdateState) =>
  state.transfer._tag === "Downloading" || state.transfer._tag === "Staging" || state.transfer._tag === "Ready"

/** A running server owns preparation; control requests never stop it. An install request follows `install`. */
export const makeHeadlessUpdateControl = (updates: ApplicationUpdate, start: Effect.Effect<void> = Effect.void, install: HeadlessInstall = packageInstallRefusal) => Effect.gen(function* () {
  // A check reports the previous update's outcome, so none runs before the service is first ready:
  // a version that never starts must report failed/startup, never applied.
  const started = yield* Deferred.make<void>()
  const schedule = yield* makeUpdateSchedule(updates.check, start.pipe(Effect.zipRight(Deferred.succeed(started, undefined)), Effect.asVoid))
  /** Checks unless an update is already offered or under way, then downloads an offer; the server installs a ready download once idle. */
  const prepareForIdleInstall = Effect.gen(function* () {
    if (!(yield* Deferred.isDone(started))) return yield* new ApplicationUpdateControlFailed({ message: "`magnitude serve` is still starting. Run `magnitude update install` again once it is ready." })
    const current = yield* updates.state
    const checked = preparing(current) || current.transfer._tag === "Available" ? current
      : yield* schedule.check.pipe(Effect.zipRight(updates.changes.pipe(Stream.filter(state => state.check._tag !== "Checking"), Stream.runHead)),
        Effect.timeout("1 minute"), Effect.orElseSucceed(() => Option.none<ApplicationUpdateState>()), Effect.map(Option.getOrElse(() => current)))
    if (checked.transfer._tag === "Available") yield* updates.download
  })
  return (action: ApplicationUpdateAction) => Effect.gen(function* () {
    if (action === "install") {
      if (install._tag === "Refuse") return yield* new ApplicationUpdateControlFailed({ message: install.message })
      yield* prepareForIdleInstall
    }
    if (action === "check" && (yield* Deferred.isDone(started))) yield* schedule.check
    if (action === "download") yield* updates.download
    if (action === "discard") yield* updates.discard
    return { state: yield* updates.state, afterReply: Effect.void }
  }).pipe(Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message })))
})

/**
 * Sends any recorded outcome with a check now and waits for its result. The check runs on the update
 * owner's worker, so an owner about to stop waits here rather than cutting the request off.
 */
export const reportUpdateOutcome = (updates: ApplicationUpdate, reason: UpdateCheckReason) => updates.check(reason).pipe(
  Effect.zipRight(updates.changes.pipe(Stream.filter(state => state.check._tag !== "Checking"), Stream.runHead)),
  Effect.timeout("15 seconds"), Effect.ignore)
