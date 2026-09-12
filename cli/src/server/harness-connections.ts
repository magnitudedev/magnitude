import { desktopIsolatedProfile, desktopDataDirectory, desktopServiceOrigin } from "./application"
import { Effect } from "effect"
import { isAbsolute, resolve } from "node:path"
import { HarnessConnectionError } from "@magnitudedev/client-common"
import { BunSqliteDriverLayer } from "@magnitudedev/daemon-management/bun"
import { makeHarnessConnectionService, harnessConnectionPaths, makeHarnessConnectorRegistry, type HarnessConnectionOptions, type HarnessConnectionPaths } from "@magnitudedev/harness-connections"
import { isDevelopmentBuild } from "../runtime/environment"
const failure = (operation: HarnessConnectionError["operation"], message: string) => new HarnessConnectionError({ operation, message })
/** One development scope shared by the launcher and every CLI it hosts. */
export const piDevelopmentConnectionOptions = (root: string): HarnessConnectionOptions & { paths: HarnessConnectionPaths } => {
  const defaults = harnessConnectionPaths(root)
  const paths = {
    ...defaults,
    manifest: resolve(root, "connections.json"),
    piModels: resolve(root, "pi/models.json"),
    piSettings: resolve(root, "pi/settings.json"),
    skillInstallations: { ...defaults.skillInstallations, "shared-agents": { skillFile: resolve(root, "skills/magnitude/SKILL.md") } },
  }
  return {
    paths,
    registry: makeHarnessConnectorRegistry(paths, { piCompanionSource: resolve(import.meta.dir, "../../../integrations/pi"), serviceEndpoint: desktopServiceOrigin }),
    serviceEndpoint: desktopServiceOrigin,
    // The desktop owns startup; development cannot register login startup.
    // Never register a development binary as the user's login service.
    installStartup: Effect.void,
  }
}

export const makeHarnessConnection = Effect.suspend(() => {
  const root = process.env.MAGNITUDE_PI_DEVELOPMENT_ROOT
  if (root === undefined) return makeHarnessConnectionService({ paths: desktopIsolatedProfile ? harnessConnectionPaths(resolve(desktopDataDirectory, "harness-home")) : harnessConnectionPaths(), serviceEndpoint: desktopServiceOrigin }).pipe(Effect.provide(BunSqliteDriverLayer))
  if (!isDevelopmentBuild() || !isAbsolute(root)) {
    return Effect.fail(failure("connect", "Pi development connection scope requires a source CLI and an absolute directory"))
  }
  return makeHarnessConnectionService(piDevelopmentConnectionOptions(root)).pipe(Effect.provide(BunSqliteDriverLayer))
})
