import { Command } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option, Schema } from "effect"
import { join } from "node:path"
import { release } from "node:os"
import { type ApplicationRuntime, type ApplicationProfile, PreparedUpdateStore, UpdatePreferences, makeUnixProcessContinuation, acquireUpdateInstallationLease,
  unixPrivateFilePermissions, windowsPrivateFilePermissions, recoverWindowsUpdateDirectory } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateSource, makeInstalledUpdatePreparation, makeApplicationUpdate,
  reconcilePreparedUpdate, unavailableApplicationUpdate, completeLinuxForegroundUpdate } from "@magnitudedev/daemon-management/application-update"
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

/** Startup installation precedes the shared package lease and every service process. */
export const prepareServeStartup = (runtime: ApplicationRuntime, profile: ApplicationProfile, addon: string, stateDirectory: string) => Effect.gen(function* () {
  if (runtime._tag !== "Installed" || process.platform !== "linux") return
  const architecture = yield* Schema.decodeUnknown(Schema.Literal("arm64", "x64"))(process.arch)
  const preparation = yield* makeInstalledUpdatePreparation({ resources: runtime.resourcesDirectory, addonPath: addon,
    dataDirectory: profile.dataDirectory, version: CLI_VERSION, osVersion: release(), platform: "linux", architecture, isolated: profile.isolated }).pipe(Effect.option)
  if (Option.isNone(preparation)) return
  const store = preparation.value.store
  const pending = yield* reconcilePreparedUpdate(CLI_VERSION).pipe(Effect.provideService(PreparedUpdateStore, store))
  if (Option.isNone(pending) || pending.value.installation._tag !== "Unattempted") return
  const authorized = yield* Command.make("/usr/bin/sudo", "-n", "-l", "--", "/usr/lib/magnitude-desktop/resources/magnitude",
    "_install-application-update", join(profile.dataDirectory, "updates", "update.json"), "--parent-stdin").pipe(Command.exitCode,
      Effect.map(code => code === 0), Effect.catchAll(() => Effect.succeed(false)))
  if (!authorized) {
    yield* Effect.sync(() => { process.stderr.write("Automatic update installation requires system authorization. Stop the server and run `magnitude update install` from a terminal to authorize it.\n") })
    return
  }
  const continuation = yield* makeUnixProcessContinuation(addon)
  yield* acquireUpdateInstallationLease(stateDirectory)
  yield* completeLinuxForegroundUpdate(profile.dataDirectory, false).pipe(Effect.provideService(PreparedUpdateStore, store))
  return yield* continuation.replace(process.execPath, process.argv.slice(2), process.env)
}).pipe(Effect.provide([unixPrivateFilePermissions.pipe(Layer.provideMerge(BunContext.layer))]))
