import { chmod, lstat, mkdir } from "node:fs/promises"
import { dirname, join } from "node:path"
import { Effect, Option, Schema } from "effect"
import { NativeHost } from "./index"
import { DataFolderUnusable, requestApplication } from "./application-control"
import { isUpdateInstallationActive } from "./update-installation-lease"

export class ApplicationOwnershipFailed extends Schema.TaggedError<ApplicationOwnershipFailed>()("ApplicationOwnershipFailed", { message: Schema.String }) {}
/** Which kind of application owns `directory`, or none when no owner answers within a second. */
export const observeApplicationOwner = (directory: string) => Effect.gen(function* () {
  const native = yield* NativeHost
  const endpoint = yield* native.inspectEndpoint(directory)
  if (Option.isNone(endpoint)) return Option.none<"Desktop" | "Headless">()
  return Option.some((yield* requestApplication(endpoint.value, "Observe")).owner._tag)
}).pipe(Effect.timeout("1 second"), Effect.catchAll(() => Effect.succeed(Option.none<"Desktop" | "Headless">())))

const applicationAlreadyRunning = (directory: string) => Effect.gen(function* () {
  const owner = yield* observeApplicationOwner(directory)
  return yield* new ApplicationOwnershipFailed({ message: Option.match(owner, {
    onNone: () => "Magnitude is already running. Stop the existing application or server before retrying.",
    onSome: owner => owner === "Desktop"
      ? "Magnitude Desktop is still running. Fully quit the desktop app before retrying."
      : "Another `magnitude serve` command is already running. Stop it before starting another.",
  }) })
})
export const ApplicationOwnerRequest = Schema.Union(
  Schema.TaggedStruct("Desktop", { intent: Schema.Literal("EnsureRunning", "ShowWindow") }),
  Schema.TaggedStruct("Headless", {}),
)
export type ApplicationOwnerRequest = typeof ApplicationOwnerRequest.Type


const USABLE_FOLDER = "Check that the folder is on a local drive you can write to."

const prepareOwnershipDirectory = (directory: string) => Effect.tryPromise({ try: async () => {
    // Windows creates the final directory with its private ACL inside native acquisition.
    await mkdir(process.platform === "win32" ? dirname(directory) : directory, { recursive: true, mode: 0o700 })
    if (process.platform !== "win32") {
      const info = await lstat(directory)
      if (!info.isDirectory() || info.isSymbolicLink() || info.uid !== process.getuid!()) throw new Error("Application directory must belong to the current user")
      await chmod(directory, 0o700)
    }
  }, catch: error => new DataFolderUnusable({
    reason: `Magnitude couldn’t prepare its data folder ${directory}. ${USABLE_FOLDER}`,
    message: `Magnitude couldn’t prepare its data folder ${directory}: ${error instanceof Error ? error.message : String(error)}`,
  }) })

/**
 * The kernel ownership lock in `directory`. Failing to open it names the folder, so the person can see which folder
 * it is (on a network drive, unwritable, or a path Windows can't open) rather than being told to reinstall.
 */
const ownershipLock = (native: NativeHost, directory: string) => native.acquireOwnership(join(directory, "application.lock")).pipe(
  Effect.mapError(error => new DataFolderUnusable({
    reason: `Magnitude couldn’t take ownership of its data folder ${directory}. ${USABLE_FOLDER}`,
    message: `Magnitude couldn’t take ownership of its data folder ${directory}: ${error.message.replace(/^Error: /, "")}`,
  })),
)

/** A handoff requests cooperation; only acquisition of the retained kernel lock admits a new owner. */
export const acquireApplicationOwner = (directory: string, request: ApplicationOwnerRequest) => Effect.gen(function* () {
  const native = yield* NativeHost
  yield* prepareOwnershipDirectory(directory)
  for (;;) {
    const lock = yield* ownershipLock(native, directory)
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

const refuseDuringUpdateInstallation = (directory: string) => Effect.gen(function* () {
  if (yield* isUpdateInstallationActive(directory)) return yield* new ApplicationOwnershipFailed({
    message: "A Magnitude update is being installed. Retry the command when it finishes.",
  })
})

/** Finite mutations use the same kernel ownership, with no control listener, takeover or service. */
export const acquireApplicationMaintenance = (directory: string) => Effect.gen(function* () {
  const native = yield* NativeHost
  yield* prepareOwnershipDirectory(directory)
  const lock = yield* ownershipLock(native, directory)
  if (Option.isNone(lock)) return yield* applicationAlreadyRunning(directory)
  yield* refuseDuringUpdateInstallation(directory)
})

/**
 * The desktop installer's ownership. A running `magnitude serve` is asked to yield, as it does when the desktop
 * opens, and serves again once the installed app has started and quit; a running desktop app is refused.
 */
export const acquireDesktopInstallation = (directory: string) => Effect.gen(function* () {
  const native = yield* NativeHost
  yield* prepareOwnershipDirectory(directory)
  for (;;) {
    const lock = yield* ownershipLock(native, directory)
    if (Option.isSome(lock)) return yield* refuseDuringUpdateInstallation(directory)
    const owner = yield* observeApplicationOwner(directory)
    if (!Option.contains(owner, "Headless")) return yield* applicationAlreadyRunning(directory)
    const endpoint = yield* native.inspectEndpoint(directory)
    if (Option.isSome(endpoint)) yield* requestApplication(endpoint.value, "Yield").pipe(Effect.ignore)
    yield* Effect.sleep("100 millis")
  }
}).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => new ApplicationOwnershipFailed({
  message: "The running `magnitude serve` has not stopped serving; the desktop app has not been installed.",
}) }))
