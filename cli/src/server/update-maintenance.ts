import { Command } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { release } from "node:os"
import { ApplicationUpdateControlFailed, type ApplicationUpdateAction } from "@magnitudedev/sdk/desktop-host"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { acquireApplicationMaintenance, acquireUpdateInstallationLease, applicationNativeHostPath, nativeHostLayer, resolveInstalledApplicationRuntime,
  PreparedUpdateStore, UpdatePreferences, unixPrivateFilePermissions, windowsPrivateFilePermissions, recoverWindowsUpdateDirectory } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateSource, makeInstalledUpdatePreparation, readPreparedUpdateState, discardPreparedUpdate, runFiniteUpdatePreparation, completeLinuxForegroundUpdate, startMacForegroundInstallation } from "@magnitudedev/daemon-management/application-update"
import { runWindowsInstalledUpdate } from "./windows-startup-update"
import { CLI_VERSION } from "../version"
import { isDevelopmentBuild } from "../runtime/environment"

/** Called only after passive owner observation reports absence; mutation replies are never replayed here. */
export const runLocalUpdateMaintenance = (options: {
  readonly action: ApplicationUpdateAction
  readonly dataDirectory: string
  readonly stateDirectory: string
  readonly isolated: boolean
}) => Effect.scoped(Effect.gen(function* () {
  if (isDevelopmentBuild()) return yield* new ApplicationUpdateControlFailed({ message: "Application updates require an installed Magnitude application." })
  const platform = yield* Schema.decodeUnknown(Schema.Literal("darwin", "linux", "win32"))(process.platform)
  const architecture = yield* Schema.decodeUnknown(Schema.Literal("arm64", "x64"))(process.arch)
  const runtime = yield* resolveInstalledApplicationRuntime(process.execPath, platform)
  const addon = applicationNativeHostPath(runtime, platform, architecture)
  const privateFiles = (platform === "win32" ? windowsPrivateFilePermissions(addon) : unixPrivateFilePermissions).pipe(Layer.provideMerge(BunContext.layer))
  const osVersion = platform === "darwin"
    ? (yield* Command.make("/usr/bin/sw_vers", "-productVersion").pipe(Command.string, Effect.timeout("5 seconds"))).trim()
    : release()
  return yield* Effect.gen(function* () {
    if (options.action === "install" && platform === "win32") yield* runWindowsInstalledUpdate(runtime, options, options.stateDirectory, false)
    else if (options.action !== "status") {
      yield* acquireApplicationMaintenance(options.stateDirectory)
      if (platform === "win32") yield* recoverWindowsUpdateDirectory(addon, options.dataDirectory)
    }
    const preparation = yield* makeInstalledUpdatePreparation({ resources: runtime.resourcesDirectory, addonPath: addon,
      dataDirectory: options.dataDirectory, version: CLI_VERSION, osVersion, platform, architecture, isolated: options.isolated })
    const execute = Effect.gen(function* () {
      if (options.action === "status") return yield* readPreparedUpdateState
      if (options.action === "install") {
        if (platform === "win32") return yield* readPreparedUpdateState
        yield* acquireUpdateInstallationLease(options.stateDirectory)
        if (platform === "darwin") return yield* startMacForegroundInstallation({ resources: runtime.resourcesDirectory,
          stateDirectory: options.stateDirectory, dataDirectory: options.dataDirectory, version: CLI_VERSION, architecture,
          operation: "Install", continuation: { _tag: "None" } })
        yield* completeLinuxForegroundUpdate(options.dataDirectory, Boolean(process.stdin.isTTY && process.stderr.isTTY))
        return yield* readPreparedUpdateState
      }
      if (options.action === "discard") return yield* discardPreparedUpdate
      const source = yield* preparation.makeSource
      return yield* runFiniteUpdatePreparation(options.action === "check" ? "check" : "download").pipe(
        Effect.provideService(ApplicationUpdateSource, source))
    })
    return yield* execute.pipe(Effect.provideService(PreparedUpdateStore, preparation.store),
      Effect.provideService(UpdatePreferences, preparation.preferences))
  }).pipe(Effect.provide([privateFiles, platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon)]))
})).pipe(Effect.provide(BunContext.layer), Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message })))
