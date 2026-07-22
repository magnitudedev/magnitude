import * as HttpServer from "@effect/platform/HttpServer"
import * as FileSystem from "@effect/platform/FileSystem"
import * as NodePath from "path"
import * as NodeOs from "os"
import { Cause, Effect, Layer, Schedule } from "effect"
import { AcnShutdown } from "./acn-shutdown"
import {
  type AcnRegistration,
  readRegistration,
  registrationIsOwnedBy,
  registrationPath,
  writeRegistrationAtomic,
} from "./daemon-registration"
import { ACN_OWNER_ID } from "./identity"
import { ACN_SHUTDOWN_TOKEN } from "./identity"

export interface DaemonLifecycleOptions {
  readonly version: string
  readonly register: boolean
  readonly debug: boolean
  readonly ownershipCheckIntervalMs?: number
  readonly ownershipCheckTimeoutMs?: number
  readonly dataDir: string
}

export const defaultDataDir = (): string => NodePath.join(NodeOs.homedir(), ".magnitude")

const getServerUrl = (address: HttpServer.Address): string => {
  if (address._tag === "UnixAddress") {
    throw new TypeError("Unix sockets are not supported for ACN registration")
  }
  return `http://${address.hostname === "0.0.0.0" ? "127.0.0.1" : address.hostname}:${address.port}`
}

export const DaemonLifecycleLive = (
  options: DaemonLifecycleOptions,
): Layer.Layer<never, never, AcnShutdown | HttpServer.HttpServer | FileSystem.FileSystem> =>
  Layer.scopedDiscard(
    Effect.gen(function* () {
      const shutdown = yield* AcnShutdown
      const server = yield* HttpServer.HttpServer
      const fs = yield* FileSystem.FileSystem
      const url = getServerUrl(server.address)
      const registrationPath_ = registrationPath(options.dataDir)
      const ownerId = ACN_OWNER_ID

      const registerDaemon = Effect.fn("acn.register")(function* () {
        const registration: AcnRegistration = {
          id: ownerId,
          version: options.version,
          url,
          pid: process.pid,
          timestamp: Date.now(),
          shutdownToken: ACN_SHUTDOWN_TOKEN,
        }
        yield* Effect.annotateCurrentSpan({
          url,
          pid: process.pid,
          version: options.version,
        })
        yield* writeRegistrationAtomic(registrationPath_, registration).pipe(
          Effect.tapError((error) =>
            Effect.logError("Failed to write ACN registration file").pipe(
              Effect.annotateLogs({ error: String(error) }),
            ),
          ),
          Effect.orDie,
        )
        yield* Effect.logInfo("ACN daemon registered").pipe(
          Effect.annotateLogs({
            url,
            id: ownerId,
            pid: process.pid,
            version: options.version,
          }),
        )
        if (options.debug) {
          yield* Effect.logDebug("ACN registered").pipe(
            Effect.annotateLogs({
              id: ownerId,
              pid: process.pid,
              version: options.version,
              url,
            }),
          )
        }
      })

      if (options.register) {
        yield* registerDaemon()
        yield* Effect.addFinalizer(() =>
          readRegistration(registrationPath_).pipe(
            Effect.flatMap((registration) =>
              registrationIsOwnedBy(registration, ownerId)
                ? fs.remove(registrationPath_, { force: true })
                : Effect.void,
            ),
            Effect.catchAll(() => Effect.void),
          ),
        )
      }

      const checkRegistration = Effect.fn("acn.check-registration")(function* () {
        const reg = yield* readRegistration(registrationPath_)
        if (registrationIsOwnedBy(reg, ownerId)) return

        yield* Effect.logWarning("ACN can no longer prove registry ownership; shutting down").pipe(
          Effect.annotateLogs({
            ownerId,
            observedOwner: reg?.id ?? null,
            observedUrl: reg?.url ?? null,
            observedPid: reg?.pid ?? null,
          }),
        )
        yield* shutdown.request({ reason: "ownership-lost" })
      })

      if (options.register) {
        yield* checkRegistration().pipe(
          Effect.timeout(`${options.ownershipCheckTimeoutMs ?? 800} millis`),
          Effect.catchAllCause((cause) =>
            Effect.logError("ACN registry ownership watchdog failed; shutting down").pipe(
              Effect.annotateLogs({ cause: Cause.pretty(cause) }),
              Effect.andThen(
                shutdown.request({
                  reason: "fatal",
                  detail: "registry ownership watchdog failed",
                }),
              ),
            ),
          ),
          Effect.repeat(Schedule.spaced(`${options.ownershipCheckIntervalMs ?? 1000} millis`)),
          Effect.forkScoped,
        )
      }
    }),
  )
