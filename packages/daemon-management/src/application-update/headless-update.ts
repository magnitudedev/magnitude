import { Effect } from "effect"
import { ApplicationUpdateControlFailed, type ApplicationUpdateAction } from "@magnitudedev/sdk/desktop-host"
import type { ApplicationUpdate } from "./application-update"
import { makeUpdateSchedule } from "./update-schedule"

/** A running server owns preparation only; no control request can install or stop it. */
export const makeHeadlessUpdateControl = (updates: ApplicationUpdate) => Effect.gen(function* () {
  const schedule = yield* makeUpdateSchedule(updates.check)
  return (action: ApplicationUpdateAction) => Effect.gen(function* () {
    if (action === "install") return yield* new ApplicationUpdateControlFailed({
      message: "Stop the server before installing an update. Then run `magnitude serve` or `magnitude update install`.",
    })
    if (action === "check") yield* schedule.check
    if (action === "download") yield* updates.download
    if (action === "discard") yield* updates.discard
    return { state: yield* updates.state, afterReply: Effect.void }
  }).pipe(Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message })))
})
