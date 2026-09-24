import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Effect, Option } from "effect"
import { applicationStateDirectory, makeDesktopApplicationHost } from "@magnitudedev/daemon-management/desktop-native"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { isDevelopmentBuild } from "../runtime/environment"

const host = makeDesktopApplicationHost(isDevelopmentBuild()
  ? Option.some(resolve(dirname(fileURLToPath(import.meta.url)), "../../.."))
  : Option.none(), bundledWindowsNative)

export const {
  desktopIsolatedProfile, desktopDataDirectory, desktopServiceOrigin, desktopApplication,
  startDesktopApplication,
  readDesktopLoginStartup, updateDesktopApplication,
} = host

export const updateApplication = (action: import("@magnitudedev/sdk/desktop-host").ApplicationUpdateAction) => Effect.gen(function* () {
  const owner = yield* host.desktopApplication.observe.pipe(Effect.map(Option.some),
    Effect.catchTag("ApplicationControlUnavailable", () => Effect.succeed(Option.none())))
  if (Option.isSome(owner)) return { owner: owner.value.owner._tag, state: yield* host.updateDesktopApplication(action) }
  const { runLocalUpdateMaintenance } = yield* Effect.promise(() => import("./update-maintenance"))
  const stateDirectory = yield* applicationStateDirectory({ platform: process.platform, dataDirectory: host.desktopDataDirectory,
    override: Option.fromNullable(process.env.MAGNITUDE_DESKTOP_STATE_DIR) })
  return { owner: "None" as const, state: yield* runLocalUpdateMaintenance({ action, stateDirectory,
    dataDirectory: host.desktopDataDirectory, isolated: host.desktopIsolatedProfile }) }
})
