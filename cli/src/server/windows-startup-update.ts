import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { release } from "node:os"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { type ApplicationRuntime, type ApplicationProfile, applicationNativeHostPath, PreparedUpdateStore,
  nativeWindowsInstallerVerifier, windowsPrivateFilePermissions } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed, completeWindowsForegroundUpdate, makeInstalledUpdatePreparation } from "@magnitudedev/daemon-management/application-update"
import { CLI_VERSION } from "../version"

export const runWindowsInstalledUpdate = (runtime: ApplicationRuntime, profile: Pick<ApplicationProfile, "dataDirectory" | "isolated">, stateDirectory: string) => Effect.gen(function* () {
  if (runtime._tag !== "Installed" || process.platform !== "win32") return false
  const addon = applicationNativeHostPath(runtime, "win32", "x64")
  return yield* Effect.gen(function* () {
    const preparation = yield* makeInstalledUpdatePreparation({ resources: runtime.resourcesDirectory, addonPath: addon,
      dataDirectory: profile.dataDirectory, version: CLI_VERSION, osVersion: release(), platform: "win32", architecture: "x64", isolated: profile.isolated })
    const publisher = preparation.configuration.windowsPublisher
    if (Option.isNone(publisher)) return yield* new ApplicationUpdateFailed({ message: "The Windows update publisher is missing." })
    return yield* completeWindowsForegroundUpdate({ resources: runtime.resourcesDirectory, dataDirectory: profile.dataDirectory,
      stateDirectory, version: CLI_VERSION }).pipe(
      Effect.provideService(PreparedUpdateStore, preparation.store), Effect.provide(nativeWindowsInstallerVerifier(addon, publisher.value)))
  }).pipe(Effect.provide([bundledWindowsNative.host, windowsPrivateFilePermissions(addon).pipe(Layer.provideMerge(BunContext.layer))]))
}).pipe(Effect.provide(BunContext.layer))
