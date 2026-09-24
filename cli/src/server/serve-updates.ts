import { Command } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Schema } from "effect"
import { release } from "node:os"
import { type ApplicationRuntime, type ApplicationProfile, PreparedUpdateStore, UpdatePreferences,
  unixPrivateFilePermissions, windowsPrivateFilePermissions, recoverWindowsUpdateDirectory } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateSource, makeInstalledUpdatePreparation, makeApplicationUpdate,
  reconcilePreparedUpdate, unavailableApplicationUpdate } from "@magnitudedev/daemon-management/application-update"
import { CLI_VERSION } from "../version"

/** Acquired by the headless owner after native admission, before it starts service work. */
export const initializeServeUpdates = (runtime: ApplicationRuntime, profile: ApplicationProfile, addon: string) => Effect.gen(function* () {
  if (runtime._tag === "Development") return unavailableApplicationUpdate("Application updates require an installed Magnitude application.")
  const platform = yield* Schema.decodeUnknown(Schema.Literal("darwin", "linux", "win32"))(process.platform)
  const architecture = yield* Schema.decodeUnknown(Schema.Literal("arm64", "x64"))(process.arch)
  const osVersion = platform === "darwin"
    ? (yield* Command.make("/usr/bin/sw_vers", "-productVersion").pipe(Command.string, Effect.timeout("5 seconds"))).trim()
    : release()
  const privateFiles = (platform === "win32" ? windowsPrivateFilePermissions(addon) : unixPrivateFilePermissions).pipe(Layer.provideMerge(BunContext.layer))
  return yield* Effect.gen(function* () {
    if (platform === "win32") yield* recoverWindowsUpdateDirectory(addon, profile.dataDirectory)
    const preparation = yield* makeInstalledUpdatePreparation({ resources: runtime.resourcesDirectory, addonPath: addon,
      dataDirectory: profile.dataDirectory, version: CLI_VERSION, osVersion, platform, architecture, isolated: profile.isolated })
    const source = yield* preparation.makeSource
    const pending = yield* reconcilePreparedUpdate(CLI_VERSION).pipe(Effect.provideService(PreparedUpdateStore, preparation.store))
    return yield* makeApplicationUpdate(pending).pipe(Effect.provideService(ApplicationUpdateSource, source),
      Effect.provideService(PreparedUpdateStore, preparation.store), Effect.provideService(UpdatePreferences, preparation.preferences))
  }).pipe(Effect.provide(privateFiles))
}).pipe(Effect.provide(BunContext.layer), Effect.catchAll(() => Effect.succeed(unavailableApplicationUpdate("Application update setup could not be read."))))
