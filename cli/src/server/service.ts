import * as Service from "@magnitudedev/daemon-management/service"
import { Effect, Option } from "effect"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { isDevelopmentBuild } from "../runtime/environment"
import { stopLocalAcn } from "./acn-instance-manager"

export const developmentServerCommand = (
  executable = process.execPath,
): ReadonlyArray<string> => [
  executable,
  resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "packages", "acn", "src", "binary.ts"),
  "serve",
]

const host = Service.ManagedServiceHost.of({
  launchCommand: isDevelopmentBuild() ? Option.some(developmentServerCommand()) : Option.none(),
  stop: stopLocalAcn,
})
const provideHost = Effect.provideService(Service.ManagedServiceHost, host)
export const installServiceOnStartup = Service.installServiceOnStartup.pipe(provideHost)
export const stopService = Service.stopService.pipe(provideHost)
export const installService = Service.installService.pipe(provideHost)
export const startInstalledService = Service.startInstalledService.pipe(provideHost)
export const uninstallService = Service.uninstallService.pipe(provideHost)
export const serviceStatus = Service.serviceStatus.pipe(provideHost)
export const startServiceManager = (...args: Parameters<typeof Service.startServiceManager>) => Service.startServiceManager(...args).pipe(provideHost)
export const confirmServicePublicReady = Service.confirmServicePublicReady.pipe(provideHost)
