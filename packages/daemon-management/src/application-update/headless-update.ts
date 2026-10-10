import { Deferred, Effect } from "effect"
import { ApplicationUpdateControlFailed, type ApplicationUpdateAction } from "@magnitudedev/sdk/desktop-host"
import type { ApplicationUpdate } from "./application-update"
import { makeUpdateSchedule } from "./update-schedule"

/** A running server owns preparation only; no control request can install or stop it. */
export const makeHeadlessUpdateControl = (updates: ApplicationUpdate, start: Effect.Effect<void> = Effect.void) => Effect.gen(function* () {
  // A check reports the previous update's outcome, so none runs before the service is first ready:
  // a version that never starts must report failed/startup, never applied.
  const started = yield* Deferred.make<void>()
  const schedule = yield* makeUpdateSchedule(updates.check, start.pipe(Effect.zipRight(Deferred.succeed(started, undefined)), Effect.asVoid))
  return (action: ApplicationUpdateAction) => Effect.gen(function* () {
    if (action === "install") return yield* new ApplicationUpdateControlFailed({
      message: "A server set up with `magnitude server setup` installs a downloaded update by itself once it is idle; restart it to install sooner. For `magnitude serve` started by hand, stop it and run `magnitude update install`.",
    })
    if (action === "check" && (yield* Deferred.isDone(started))) yield* schedule.check
    if (action === "download") yield* updates.download
    if (action === "discard") yield* updates.discard
    return { state: yield* updates.state, afterReply: Effect.void }
  }).pipe(Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message })))
})
