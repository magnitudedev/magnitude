import { chmod, lstat, mkdir } from "node:fs/promises"
import { dirname, join } from "node:path"
import { Effect, Option, Schedule, Schema } from "effect"
import { NativeHost } from "./index"
import { requestApplication, type ApplicationIntent } from "./application-control"

export class ApplicationOwnershipFailed extends Schema.TaggedError<ApplicationOwnershipFailed>()("ApplicationOwnershipFailed", { message: Schema.String }) {}
/** The directory and lock are stable across versions. No PID election, unlink, or takeover. */
export const acquireApplicationOwner = (directory: string, intent: ApplicationIntent) => Effect.gen(function* () {
  const native = yield* NativeHost
  yield* Effect.tryPromise({ try: async () => {
    // Windows creates the final directory with its private ACL inside native acquisition.
    await mkdir(process.platform === "win32" ? dirname(directory) : directory, { recursive: true, mode: 0o700 })
    if (process.platform !== "win32") {
      const info = await lstat(directory)
      if (!info.isDirectory() || info.isSymbolicLink() || info.uid !== process.getuid!()) throw new Error("Application directory must belong to the current user")
      await chmod(directory, 0o700)
    }
  }, catch: error => new ApplicationOwnershipFailed({ message: String(error) }) })
  const lock = yield* native.acquireOwnership(join(directory, "application.lock"))
  if (Option.isSome(lock)) return { _tag: "Owner" as const, socketPath: yield* native.ownedEndpoint(lock.value, directory), lock: lock.value }
  const endpoint = yield* native.inspectEndpoint(directory)
  if (Option.isNone(endpoint)) return yield* new ApplicationOwnershipFailed({ message: "The existing application directory could not be resolved." })
  const socketPath = endpoint.value
  // A cold owner may still be binding its socket. Retrying intent is harmless and never takes ownership.
  const response = yield* requestApplication(socketPath, intent).pipe(Effect.retry({ times: 20, schedule: Schedule.spaced("100 millis") }), Effect.timeoutFail({ duration: "5 seconds", onTimeout: () => new ApplicationOwnershipFailed({ message: "The existing application did not respond; it has not been replaced" }) }))
  return { _tag: "Forwarded" as const, response }
})
