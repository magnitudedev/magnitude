import { chmod, lstat, mkdir } from "node:fs/promises"
import { dirname, join } from "node:path"
import { Effect, Option, Schema } from "effect"
import { NativeHost } from "./index"
import { requestApplication } from "./application-control"

export class ApplicationOwnershipFailed extends Schema.TaggedError<ApplicationOwnershipFailed>()("ApplicationOwnershipFailed", { message: Schema.String }) {}
export const ApplicationOwnerRequest = Schema.Union(
  Schema.TaggedStruct("Desktop", { intent: Schema.Literal("EnsureRunning", "ShowWindow") }),
  Schema.TaggedStruct("Headless", {}),
)
export type ApplicationOwnerRequest = typeof ApplicationOwnerRequest.Type

/** A handoff requests cooperation; only acquisition of the retained kernel lock admits a new owner. */
export const acquireApplicationOwner = (directory: string, request: ApplicationOwnerRequest) => Effect.gen(function* () {
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
  for (;;) {
    const lock = yield* native.acquireOwnership(join(directory, "application.lock"))
    if (Option.isSome(lock)) return { _tag: "Owner" as const, socketPath: yield* native.ownedEndpoint(lock.value, directory), lock: lock.value }
    if (request._tag === "Headless") return yield* new ApplicationOwnershipFailed({ message: "Magnitude is already running. Stop the existing service before running `magnitude serve`." })
    const endpoint = yield* native.inspectEndpoint(directory)
    if (Option.isNone(endpoint)) return yield* new ApplicationOwnershipFailed({ message: "The existing application directory could not be resolved." })
    const socketPath = endpoint.value
    // A cold owner may still be binding its endpoint. Malformed replies are never absence.
    const observed = yield* requestApplication(socketPath, "Observe").pipe(
      Effect.map(Option.some), Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())),
    )
    if (Option.isNone(observed)) { yield* Effect.sleep("100 millis"); continue }
    const response = observed.value
    const forwarded = yield* requestApplication(socketPath, response.owner._tag === "Desktop" ? request.intent : "Yield").pipe(
      Effect.map(Option.some), Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())),
    )
    if (Option.isSome(forwarded) && forwarded.value.owner._tag === "Desktop" && response.owner._tag === "Desktop") {
      return { _tag: "Forwarded" as const, response: forwarded.value }
    }
    // Yield acknowledgement precedes shutdown. It does not prove retirement or transfer the lock.
    yield* Effect.sleep("100 millis")
  }
}).pipe(Effect.timeoutFail({ duration: "60 seconds", onTimeout: () => new ApplicationOwnershipFailed({
  message: "The existing Magnitude owner has not finished stopping; it has not been replaced.",
}) }))
