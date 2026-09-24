import { BunContext } from "@effect/platform-bun"
import { Effect, Either, Layer, Option } from "effect"
import { release } from "node:os"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { type ApplicationRuntime, type ApplicationProfile, applicationNativeHostPath, PreparedUpdateStore,
  nativeWindowsInstallerVerifier, windowsPrivateFilePermissions } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed, completeWindowsForegroundUpdate, makeInstalledUpdatePreparation } from "@magnitudedev/daemon-management/application-update"
import { CLI_VERSION } from "../version"

export const runWindowsInstalledUpdate = (runtime: ApplicationRuntime, profile: Pick<ApplicationProfile, "dataDirectory" | "isolated">, stateDirectory: string, automatic: boolean) => Effect.gen(function* () {
  if (runtime._tag !== "Installed" || process.platform !== "win32") return false
  const addon = applicationNativeHostPath(runtime, "win32", "x64")
  return yield* Effect.gen(function* () {
    const initialized = yield* makeInstalledUpdatePreparation({ resources: runtime.resourcesDirectory, addonPath: addon,
      dataDirectory: profile.dataDirectory, version: CLI_VERSION, osVersion: release(), platform: "win32", architecture: "x64", isolated: profile.isolated }).pipe(Effect.either)
    if (Either.isLeft(initialized)) {
      if (automatic) return false
      return yield* initialized.left
    }
    const preparation = initialized.right
    const publisher = preparation.configuration.windowsPublisher
    if (Option.isNone(publisher)) {
      if (automatic) return false
      return yield* new ApplicationUpdateFailed({ message: "The Windows update publisher is missing." })
    }
    return yield* completeWindowsForegroundUpdate({ resources: runtime.resourcesDirectory, dataDirectory: profile.dataDirectory,
      stateDirectory, version: CLI_VERSION, automatic, launcherProtocol: process.env.MAGNITUDE_CLI_LAUNCHER_PROTOCOL }).pipe(
      Effect.provideService(PreparedUpdateStore, preparation.store), Effect.provide(nativeWindowsInstallerVerifier(addon, publisher.value)))
  }).pipe(Effect.provide([bundledWindowsNative.host, windowsPrivateFilePermissions(addon).pipe(Layer.provideMerge(BunContext.layer))]))
}).pipe(Effect.provide(BunContext.layer))
