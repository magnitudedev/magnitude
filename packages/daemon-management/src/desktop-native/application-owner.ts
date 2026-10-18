import { chmod, lstat, mkdir } from "node:fs/promises"
import { dirname, join } from "node:path"
import { Effect, Option, Schema } from "effect"
import { NativeHost } from "./index"
import { requestApplication } from "./application-control"
import { ApplicationOpenUrl } from "@magnitudedev/sdk/desktop-host"
import { isUpdateInstallationActive } from "./update-installation-lease"
import { occupiedMessage } from "./service-port"

export class ApplicationOwnershipFailed extends Schema.TaggedError<ApplicationOwnershipFailed>()("ApplicationOwnershipFailed", { message: Schema.String }) {}
export class ApplicationServiceRunning extends Schema.TaggedError<ApplicationServiceRunning>()("ApplicationServiceRunning", { message: Schema.String }) {}
const applicationAlreadyRunning = (directory: string) => Effect.gen(function* () {
  const owner = yield* Effect.gen(function* () {
    const native = yield* NativeHost
    const endpoint = yield* native.inspectEndpoint(directory)
    if (Option.isNone(endpoint)) return Option.none()
    return Option.some((yield* requestApplication(endpoint.value, "Observe")).owner._tag)
  }).pipe(Effect.timeout("1 second"), Effect.catchAll(() => Effect.succeed(Option.none())))
  return yield* new ApplicationOwnershipFailed({ message: Option.match(owner, {
    onNone: () => "Magnitude is already running. Stop the existing application or server before retrying.",
    onSome: owner => owner === "Desktop"
      ? "Magnitude Desktop is still running. Fully quit the desktop app before retrying."
      : "Another `magnitude serve` command is already running. Stop it before starting another.",
  }) })
})
export const ApplicationOwnerRequest = Schema.Union(
  Schema.TaggedStruct("Desktop", { intent: Schema.Union(Schema.Literal("EnsureRunning", "ShowWindow"), ApplicationOpenUrl) }),
  Schema.TaggedStruct("Headless", {}),
)
export type ApplicationOwnerRequest = typeof ApplicationOwnerRequest.Type

/**
 * Private profiles keep the ownership directory owner-only. The server profile lets every local
 * account traverse it to reach the control socket, without listing it or opening the lock.
 */
const ownershipDirectoryMode = (sharedControl: boolean) => sharedControl ? 0o711 : 0o700

const prepareOwnershipDirectory = (directory: string, sharedControl = false) => Effect.tryPromise({ try: async () => {
    // Windows creates the final directory with its private ACL inside native acquisition.
    await mkdir(process.platform === "win32" ? dirname(directory) : directory, { recursive: true, mode: ownershipDirectoryMode(sharedControl) })
    if (process.platform !== "win32") {
      const info = await lstat(directory)
      if (!info.isDirectory() || info.isSymbolicLink() || info.uid !== process.getuid!()) throw new Error("Application directory must belong to the current user")
      await chmod(directory, ownershipDirectoryMode(sharedControl))
    }
  }, catch: error => new ApplicationOwnershipFailed({ message: String(error) }) })

/** A handoff requests cooperation; only acquisition of the retained kernel lock admits a new owner. */
export const acquireApplicationOwner = (directory: string, request: ApplicationOwnerRequest, sharedControl = false) => Effect.gen(function* () {
  const native = yield* NativeHost
  yield* prepareOwnershipDirectory(directory, sharedControl)
  for (;;) {
    const lock = yield* native.acquireOwnership(join(directory, "application.lock"))
    if (Option.isSome(lock)) return { _tag: "Owner" as const, socketPath: yield* native.ownedEndpoint(lock.value, directory), lock: lock.value }
    if (request._tag === "Headless") return yield* applicationAlreadyRunning(directory)
    const endpoint = yield* native.inspectEndpoint(directory)
    if (Option.isNone(endpoint)) return yield* new ApplicationOwnershipFailed({ message: "The existing application directory could not be resolved." })
    const socketPath = endpoint.value
    // A cold owner may still be binding its endpoint. Malformed replies are never absence.
    const observed = yield* requestApplication(socketPath, "Observe").pipe(
      Effect.map(Option.some), Effect.catchTags({ ApplicationControlUnavailable: () => Effect.succeed(Option.none()),
        ApplicationControlClosed: () => Effect.succeed(Option.none()) }),
    )
    if (Option.isNone(observed)) { yield* Effect.sleep("100 millis"); continue }
    const response = observed.value
    // A service manager would restart a service the desktop stopped, so only a terminal serve yields.
    if (response.owner._tag === "Headless" && response.owner.service) {
      return yield* new ApplicationServiceRunning({ message: occupiedMessage(Number(new URL(response.endpoint).port), true) })
    }
    const forwarded = yield* requestApplication(socketPath, response.owner._tag === "Desktop" ? request.intent : "Yield").pipe(
      Effect.map(Option.some), Effect.catchTags({ ApplicationControlUnavailable: () => Effect.succeed(Option.none()),
        ApplicationControlClosed: () => Effect.succeed(Option.none()) }),
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

/** Finite mutations use the same kernel ownership, with no control listener, takeover or service. */
export const acquireApplicationMaintenance = (directory: string, sharedControl = false) => Effect.gen(function* () {
  const native = yield* NativeHost
  yield* prepareOwnershipDirectory(directory, sharedControl)
  const lock = yield* native.acquireOwnership(join(directory, "application.lock"))
  if (Option.isNone(lock)) return yield* applicationAlreadyRunning(directory)
  if (yield* isUpdateInstallationActive(directory)) return yield* new ApplicationOwnershipFailed({
    message: "A Magnitude update is being installed. Retry the command when it finishes.",
  })
})
