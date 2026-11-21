import { Command } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { release } from "node:os"
import { ApplicationUpdateControlFailed, type ApplicationUpdateAction } from "@magnitudedev/sdk/desktop-host"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { acquireApplicationMaintenance, acquireUpdateInstallationLease, isUpdateInstallationActive, applicationNativeHostPath, nativeHostLayer, resolveInstalledApplicationRuntime,
  PreparedUpdateStore, UpdatePreferences, standaloneInstallationRoot, unixPrivateFilePermissions, windowsPrivateFilePermissions, recoverWindowsUpdateDirectory, nodeTerminalCommand } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateSource, makeInstalledUpdatePreparation, readPreparedUpdateState, discardPreparedUpdate, runFiniteUpdatePreparation, completeLinuxForegroundUpdate, startMacForegroundInstallation, UpdateHardwareSource, unavailableUpdateHardware,
  completeStandaloneUpdate, nativeStandaloneCodeVerifier } from "@magnitudedev/daemon-management/application-update"
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
    const standalone = runtime.layout === "Standalone"
    if (options.action === "install" && platform === "win32" && !standalone) yield* runWindowsInstalledUpdate(runtime, options, options.stateDirectory)
    else if (options.action !== "status") {
      yield* acquireApplicationMaintenance(options.stateDirectory)
      if (platform === "win32") yield* recoverWindowsUpdateDirectory(addon, options.dataDirectory)
    }
    const preparation = yield* makeInstalledUpdatePreparation({ resources: runtime.resourcesDirectory, layout: runtime.layout, addonPath: addon,
      dataDirectory: options.dataDirectory, version: CLI_VERSION, osVersion, platform, architecture, isolated: options.isolated, owner: "headless" })
    const prepare = (action: "check" | "download") => preparation.makeSource.pipe(Effect.provideService(UpdateHardwareSource, unavailableUpdateHardware),
      Effect.flatMap(source => runFiniteUpdatePreparation(action).pipe(Effect.provideService(ApplicationUpdateSource, source))))
    const execute = Effect.gen(function* () {
      if (options.action === "status") {
        const state = yield* readPreparedUpdateState
        // While the installer runs, its record already says it was attempted; that is not yet a failure.
        return (yield* isUpdateInstallationActive(options.stateDirectory))
          ? { ...state, transfer: { _tag: "Unavailable", kind: "Installing", message: "An update is being installed.\nCheck again when it finishes: magnitude update status" } } as const
          : state
      }
      if (options.action === "install") {
        if (standalone) {
          // On demand, a CLI-only installation downloads the newest release first when nothing is prepared.
          const prepared = yield* readPreparedUpdateState
          if (prepared.transfer._tag !== "Ready" && prepared.transfer._tag !== "InstallationFailed") {
            const downloaded = yield* prepare("download")
            if (downloaded.transfer._tag !== "Ready") return yield* new ApplicationUpdateControlFailed({ message: "Magnitude is up to date." })
          }
          yield* acquireUpdateInstallationLease(options.stateDirectory)
          yield* completeStandaloneUpdate({ root: standaloneInstallationRoot(runtime, platform), runningVersion: CLI_VERSION, platform, environment: process.env }).pipe(
            Effect.provide(nativeStandaloneCodeVerifier(platform, addon, preparation.configuration.windowsPublisher)))
          return yield* readPreparedUpdateState
        }
        if (platform === "win32") return yield* readPreparedUpdateState
        yield* acquireUpdateInstallationLease(options.stateDirectory)
        if (platform === "darwin") return yield* startMacForegroundInstallation({ resources: runtime.resourcesDirectory,
          stateDirectory: options.stateDirectory, dataDirectory: options.dataDirectory, version: CLI_VERSION, architecture,
          operation: "Install", continuation: { _tag: "None" } })
        yield* completeLinuxForegroundUpdate(options.dataDirectory, process.stdin.isTTY && process.stderr.isTTY ? "Terminal" : "Unattended").pipe(Effect.provide(nodeTerminalCommand))
        return yield* readPreparedUpdateState
      }
      if (options.action === "discard") return yield* discardPreparedUpdate
      return yield* prepare(options.action === "check" ? "check" : "download")
    })
    return yield* execute.pipe(Effect.provideService(PreparedUpdateStore, preparation.store),
      Effect.provideService(UpdatePreferences, preparation.preferences))
  }).pipe(Effect.provide([privateFiles, platform === "win32" ? bundledWindowsNative.host : nativeHostLayer(addon)]))
})).pipe(Effect.provide(BunContext.layer), Effect.mapError(error => new ApplicationUpdateControlFailed({ message: error.message })))
