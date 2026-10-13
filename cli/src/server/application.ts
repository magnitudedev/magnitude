import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Layer, Option, Schema } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { MagnitudeClient, ServiceAccessDenied } from "@magnitudedev/sdk"
import { applicationStateDirectory, makeDesktopApplicationHost } from "@magnitudedev/daemon-management/desktop-native"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { isDevelopmentBuild } from "../runtime/environment"

const host = makeDesktopApplicationHost(isDevelopmentBuild()
  ? Option.some(resolve(dirname(fileURLToPath(import.meta.url)), "../../.."))
  : Option.none(), bundledWindowsNative)

export const {
  desktopIsolatedProfile, desktopDataDirectory, desktopServiceOrigin, desktopApplication, serverProfile, applicationDataDirectory,
  startDesktopApplication,
  readDesktopLoginStartup, updateDesktopApplication,
} = host

/**
 * The SDK client for the service on this machine's loopback port, without starting it. It presents
 * the local control token where the service requires one.
 */
export const localServiceClient = (origin: string) => MagnitudeClient.layer({ origin, autoStart: false,
  localControl: host.localControlToken.pipe(Effect.mapError(error => new ServiceAccessDenied({ origin, message: error.message }))),
}).pipe(Layer.provide(FetchHttpClient.layer))

export class ServerNotRunning extends Schema.TaggedError<ServerNotRunning>()("ServerNotRunning", {}) {
  override get message() { return "The Magnitude server isn't running. Start it with `sudo systemctl start magnitude`, and see its log with `journalctl -u magnitude`." }
}

export const updateApplication = (action: import("@magnitudedev/sdk/desktop-host").ApplicationUpdateAction) => Effect.gen(function* () {
  const owner = yield* host.desktopApplication.observe.pipe(Effect.map(Option.some),
    Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())))
  if (Option.isSome(owner)) return { owner: owner.value.owner._tag, state: yield* host.updateDesktopApplication(action) }
  // The service owns the server profile's updates; another account cannot maintain its files.
  if (yield* host.serverProfile) return yield* new ServerNotRunning()
  const { runLocalUpdateMaintenance } = yield* Effect.promise(() => import("./update-maintenance"))
  const stateDirectory = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: host.desktopDataDirectory,
    override: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_STATE_DIR) })
  return { owner: "None" as const, state: yield* runLocalUpdateMaintenance({ action, stateDirectory,
    dataDirectory: host.desktopDataDirectory, isolated: host.desktopIsolatedProfile }) }
})
