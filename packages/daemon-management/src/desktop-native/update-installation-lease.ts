import { Effect, Option } from "effect"
import { existsSync } from "node:fs"
import { join } from "node:path"
import { NativeHost, NativeHostUnavailable } from "./index"

/** A live helper holds this kernel lease across owner exit, installer execution and result recording. */
export const acquireUpdateInstallationLease = (stateDirectory: string) => Effect.gen(function* () {
  const native = yield* NativeHost
  const lease = yield* native.acquireOwnership(join(stateDirectory, "update-installation.lock"))
  if (Option.isNone(lease)) return yield* new NativeHostUnavailable({ message: "An application update is already being installed." })
})

/**
 * Do not wait while holding application ownership: an installer may need that ownership to proceed. An installer
 * creates the lock file before holding it, so without one (as in a state folder not yet created) none is running,
 * and the probe creates nothing.
 */
export const isUpdateInstallationActive = (stateDirectory: string) => Effect.scoped(Effect.gen(function* () {
  const lock = join(stateDirectory, "update-installation.lock")
  if (!(yield* Effect.sync(() => existsSync(lock)))) return false
  const native = yield* NativeHost
  return Option.isNone(yield* native.acquireOwnership(lock))
}))
