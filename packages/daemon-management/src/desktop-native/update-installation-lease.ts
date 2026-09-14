import { Effect, Option } from "effect"
import { join } from "node:path"
import { NativeHost, NativeHostUnavailable } from "./index"

/** A live helper holds this kernel lease across owner exit, installer execution and result recording. */
export const acquireUpdateInstallationLease = (stateDirectory: string) => Effect.gen(function* () {
  const native = yield* NativeHost
  const lease = yield* native.acquireOwnership(join(stateDirectory, "update-installation.lock"))
  if (Option.isNone(lease)) return yield* new NativeHostUnavailable({ message: "An application update is already being installed." })
})

/** Do not wait while holding application ownership: an installer may need that ownership to proceed. */
export const isUpdateInstallationActive = (stateDirectory: string) => Effect.scoped(Effect.gen(function* () {
  const native = yield* NativeHost
  return Option.isNone(yield* native.acquireOwnership(join(stateDirectory, "update-installation.lock")))
}))
