import { Effect, Option } from "effect"
import { LegacyMigrationFailed } from "./legacy-migration"
import type { LegacyOwner, LegacyOwnerInvalid, LegacyOwnerReadFailed } from "./legacy-owner"
import type { LegacyStartupFailed } from "./legacy-startup-command"
import type { LegacyWindowsStartup } from "./legacy-startup-windows"
import { legacyWindowsProcessIdentity } from "./legacy-windows-owner"

/** Existing installations must enter exact migration; absence alone permits fresh installation. */
export const requireFreshWindowsInstallation = (observations: {
  readonly owner: Effect.Effect<Option.Option<LegacyOwner>, LegacyOwnerInvalid | LegacyOwnerReadFailed>
  readonly task: Effect.Effect<Option.Option<LegacyWindowsStartup>, LegacyStartupFailed>
}) => Effect.gen(function* () {
  const owner = yield* observations.owner
  if (Option.isSome(owner)) yield* legacyWindowsProcessIdentity(owner.value)
  const task = yield* observations.task
  if (Option.isSome(owner) || Option.isSome(task)) {
    return yield* new LegacyMigrationFailed({
      message: "An existing Windows Magnitude service requires migration before this app can start inference. Windows service migration is not implemented in this build.",
    })
  }
}).pipe(Effect.mapError(error => error._tag === "LegacyMigrationFailed" ? error : new LegacyMigrationFailed({ message: error.message })))
