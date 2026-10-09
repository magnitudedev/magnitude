import { Effect } from "effect"
import { ApplicationUpdateControlFailed, type ApplicationUpdateAction } from "@magnitudedev/sdk/desktop-host"
import type { ApplicationUpdate } from "./application-update"
import { makeUpdateSchedule } from "./update-schedule"

/** A running server owns preparation only; no control request can install or stop it. */
export const makeHeadlessUpdateControl = (updates: ApplicationUpdate, start?: Effect.Effect<void>) => Effect.gen(function* () {
  const schedule = yield* makeUpdateSchedule(updates.check, start)
  return (action: ApplicationUpdateAction) => Effect.gen(function* () {
    if (action === "install") return yield* new ApplicationUpdateControlFailed({
      message: "The server installs a downloaded update by itself once it is idle. To install now, stop it and run `magnitude update install`.",
    })
    if (action === "check") yield* schedule.check
    if (action === "download") yield* updates.download
    if (action === "discard") yield* updates.discard
    return { state: yield* updates.state, afterReply: Effect.void }
  }).pipe(Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message })))
})
