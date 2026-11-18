import { Effect, Option } from "effect"
import { observeApplicationOwner } from "./application-owner"
import { portServedByOtherMagnitude } from "./service-port"

/** What a `serve` waits for before it may own its data folder and port. */
export type ServingBlocker = "Desktop" | "AnotherMagnitude"

/**
 * The desktop app owning this data folder, or another Magnitude (another account's, or one with another
 * data folder) serving the port. A second `serve` in the same folder is not a reason to wait; owner
 * admission refuses it.
 */
export const servingBlocker = (stateDirectory: string, port: number) => Effect.gen(function* () {
  const owner = yield* observeApplicationOwner(stateDirectory)
  if (Option.contains(owner, "Desktop")) return Option.some<ServingBlocker>("Desktop")
  if (Option.isSome(owner)) return Option.none<ServingBlocker>()
  return (yield* portServedByOtherMagnitude(port)) ? Option.some<ServingBlocker>("AnotherMagnitude") : Option.none<ServingBlocker>()
})
