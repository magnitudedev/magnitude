import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Clock, Context, Duration, Effect, Layer, Option, Ref, Schema } from "effect"
import { dirname, join } from "node:path"
import { release } from "node:os"
import type { UpdateOwner } from "@magnitudedev/release/hosted-update"
import { type ApplicationRuntime, type ApplicationProfile, PreparedUpdateStore, UpdatePreferences, makeUnixProcessContinuation, acquireUpdateInstallationLease,
  acquireApplicationMaintenance, nativeHostLayer, unixPrivateFilePermissions, windowsPrivateFilePermissions, recoverWindowsUpdateDirectory, nodeTerminalCommand, linuxInstallationLockHeldByOthers } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateSource, makeInstalledUpdatePreparation, makeApplicationUpdate,
  reconcilePreparedUpdate, unavailableApplicationUpdate, completeLinuxForegroundUpdate, prepareMacForegroundStartup,
  startMacForegroundInstallation } from "@magnitudedev/daemon-management/application-update"
import { CLI_VERSION } from "../version"
import { withLocalService } from "./server-reach"

const linuxInstalledCli = "/usr/lib/magnitude-desktop/resources/magnitude"
const linuxInstallationLock = "/var/lib/magnitude-desktop/installation.lock"
/** How long a deferred installation waits before trying again at the next idle point. */
const deferredRetry = Duration.minutes(15)
const idlePoll = Duration.seconds(10)

const architectureOf = Schema.decodeUnknown(Schema.Literal("arm64", "x64"))
const osVersion = (platform: "darwin" | "linux" | "win32") => platform === "darwin"
  ? Command.make("/usr/bin/sw_vers", "-productVersion").pipe(Command.string, Effect.map(value => value.trim()), Effect.timeout("5 seconds"))
  : Effect.succeed(release())

const preparation = (runtime: Extract<ApplicationRuntime, { _tag: "Installed" }>, profile: ApplicationProfile, addon: string, owner: UpdateOwner) => Effect.gen(function* () {
  const platform = yield* Schema.decodeUnknown(Schema.Literal("darwin", "linux", "win32"))(process.platform)
  return yield* makeInstalledUpdatePreparation({ resources: runtime.resourcesDirectory, addonPath: addon,
    dataDirectory: profile.dataDirectory, version: CLI_VERSION, osVersion: yield* osVersion(platform), platform,
    architecture: yield* architectureOf(process.arch), isolated: profile.isolated, owner })
})

/** Acquired by the headless owner after native admission, before it starts service work. */
export const initializeServeUpdates = (runtime: ApplicationRuntime, profile: ApplicationProfile, addon: string, owner: UpdateOwner) => Effect.gen(function* () {
  if (runtime._tag === "Development") return unavailableApplicationUpdate("Application updates require an installed Magnitude application.")
  const privateFiles = (process.platform === "win32" ? windowsPrivateFilePermissions(addon) : unixPrivateFilePermissions).pipe(Layer.provideMerge(BunContext.layer))
  return yield* Effect.gen(function* () {
    if (process.platform === "win32") yield* recoverWindowsUpdateDirectory(addon, profile.dataDirectory)
    const prepared = yield* preparation(runtime, profile, addon, owner)
    const source = yield* prepared.makeSource
    const pending = yield* reconcilePreparedUpdate(CLI_VERSION).pipe(Effect.provideService(PreparedUpdateStore, prepared.store))
    return yield* makeApplicationUpdate(pending).pipe(Effect.provideService(ApplicationUpdateSource, source),
      Effect.provideService(PreparedUpdateStore, prepared.store), Effect.provideService(UpdatePreferences, prepared.preferences))
  }).pipe(Effect.provide(privateFiles))
}).pipe(Effect.provide(BunContext.layer), Effect.catchAll(() => Effect.succeed(unavailableApplicationUpdate("Application update setup could not be read."))))

/** Startup only completes an interrupted macOS transaction; downloaded updates install at an idle point. */
export const prepareServeStartup = (runtime: ApplicationRuntime, profile: ApplicationProfile, stateDirectory: string) => Effect.gen(function* () {
  if (runtime._tag !== "Installed" || process.platform !== "darwin") return
  return yield* prepareMacForegroundStartup({ resources: runtime.resourcesDirectory,
    stateDirectory, dataDirectory: profile.dataDirectory, version: CLI_VERSION, architecture: yield* architectureOf(process.arch), arguments: process.argv.slice(2) })
}).pipe(Effect.provide([unixPrivateFilePermissions.pipe(Layer.provideMerge(BunContext.layer))]))

/** True once the service reports no working sessions and no inference requests in flight. */
const isIdle = (endpoint: string) => withLocalService(endpoint, client => client.application.getServiceActivity({})).pipe(
  Effect.map(activity => activity.workingSessions === 0 && activity.inferenceRequests === 0), Effect.orElseSucceed(() => false))

/** The machine an idle installation acts on; replaced by fakes to exercise the decisions. */
export interface IdleInstallationSystem {
  /** Whether this process may install without asking anyone: the service's sudoers rule, or a writable app folder. */
  readonly canInstallUnattended: Effect.Effect<boolean>
  readonly isIdle: Effect.Effect<boolean>
  /** False while another Magnitude holds the installation lock the package checks. */
  readonly installationLockFree: Effect.Effect<boolean>
  readonly store: Effect.Effect<PreparedUpdateStore, { readonly message: string }>
  /** Admission, the platform installer and, on success, replacement of this process; returns only on failure. */
  readonly installPrepared: Effect.Effect<void, { readonly message: string }>
  readonly notify: (line: string) => Effect.Effect<void>
}
export const IdleInstallationSystem = Context.GenericTag<IdleInstallationSystem>("@magnitudedev/cli/IdleInstallationSystem")

/**
 * Install when idle: a running server installs a downloaded update only between turns. Without
 * unattended authorization it keeps serving and says how to install; a held installation lock defers.
 */
export const makeIdleInstallation = Effect.gen(function* () {
  const system = yield* IdleInstallationSystem
  const retryAfter = yield* Ref.make(0)
  const noticed = yield* Ref.make(new Set<string>())

  const defer = (version: string) => Effect.gen(function* () {
    yield* (yield* system.store).recordOutcome({ outcome: "deferred", version, reason: Option.none() })
    yield* Ref.set(retryAfter, (yield* Clock.currentTimeMillis) + Duration.toMillis(deferredRetry))
    yield* system.notify(`Magnitude ${version} is waiting for another Magnitude to quit; trying again later.`)
  })

  /**
   * Resolves true at an idle point where the installation lock is free, so the caller may stop and
   * install. The package refuses to upgrade while anyone else holds that lock, so a held lock defers
   * without stopping: the deferral is reported through `report` and the current version keeps serving.
   */
  const installWhenIdle = (version: string, report: Effect.Effect<void> = Effect.void) => Effect.gen(function* () {
    if (!(yield* system.canInstallUnattended)) {
      const first = !(yield* Ref.get(noticed)).has(version)
      yield* Ref.update(noticed, seen => new Set([...seen, version]))
      if (first) yield* system.notify(`Magnitude ${version} is downloaded. Install it with \`magnitude update install\`.`)
      return false
    }
    while (true) {
      const wait = (yield* Ref.get(retryAfter)) - (yield* Clock.currentTimeMillis)
      if (wait > 0) yield* Effect.sleep(Duration.millis(wait))
      while (!(yield* system.isIdle)) yield* Effect.sleep(idlePoll)
      if (yield* system.installationLockFree) return true
      yield* defer(version).pipe(Effect.zipRight(report), Effect.catchAll(() => Effect.void))
    }
  })

  const install = Effect.gen(function* () {
    const store = yield* system.store
    const pending = yield* store.read
    if (Option.isNone(pending)) return
    const version = pending.value.release.version
    // The lock may have been taken since the idle decision.
    if (!(yield* system.installationLockFree)) return yield* defer(version)
    yield* system.installPrepared
  }).pipe(Effect.catchAll(error => Effect.gen(function* () {
    // Unattended, a failed download would hold back every later release: keep only its outcome, so
    // the next check can fetch a newer version (or this one again, unless it was withdrawn).
    const store = yield* system.store
    const failed = yield* store.read
    if (Option.isSome(failed) && failed.value.installation._tag === "Failed") {
      yield* store.recordOutcome({ outcome: "failed", version: failed.value.release.version, reason: Option.some(failed.value.installation.kind) })
      yield* store.discard
    }
    yield* system.notify(`The update could not be installed: ${error.message} Serving the current version.`)
  }).pipe(Effect.catchAll(() => system.notify(`The update could not be installed: ${error.message} Serving the current version.`)))))

  const startupFailed = system.store.pipe(Effect.flatMap(store => Effect.gen(function* () {
    const outcome = yield* store.outcome
    if (Option.isSome(outcome) && outcome.value.outcome === "applied" && outcome.value.version === CLI_VERSION) {
      yield* store.recordOutcome({ outcome: "failed", version: CLI_VERSION, reason: Option.some("startup") })
    }
  })), Effect.ignore)

  return { installWhenIdle, install, startupFailed }
})

export const idleInstallationSystem = (options: {
  readonly runtime: Extract<ApplicationRuntime, { _tag: "Installed" }>
  readonly profile: ApplicationProfile
  readonly stateDirectory: string
  readonly addon: string
  readonly owner: UpdateOwner
  readonly notify: (line: string) => Effect.Effect<void>
}) => Layer.effect(IdleInstallationSystem, Effect.gen(function* () {
  const executor = yield* CommandExecutor.CommandExecutor
  const fs = yield* FileSystem.FileSystem
  const privateFiles = unixPrivateFilePermissions.pipe(Layer.provideMerge(BunContext.layer))
  const store = preparation(options.runtime, options.profile, options.addon, options.owner).pipe(
    Effect.map(prepared => prepared.store), Effect.provide(privateFiles))
  const installLinux = Effect.scoped(Effect.gen(function* () {
    yield* acquireApplicationMaintenance(options.stateDirectory, options.profile.groupAccess)
    const continuation = yield* makeUnixProcessContinuation(options.addon)
    yield* acquireUpdateInstallationLease(options.stateDirectory)
    const version = yield* completeLinuxForegroundUpdate(options.profile.dataDirectory, false).pipe(Effect.provideServiceEffect(PreparedUpdateStore, store), Effect.provide(nodeTerminalCommand))
    yield* options.notify(`Installed Magnitude ${version}. Restarting.`)
    return yield* continuation.replace(process.execPath, process.argv.slice(2), process.env)
  }))
  const installMac = Effect.scoped(Effect.gen(function* () {
    yield* acquireApplicationMaintenance(options.stateDirectory)
    yield* acquireUpdateInstallationLease(options.stateDirectory)
    return yield* startMacForegroundInstallation({ resources: options.runtime.resourcesDirectory, stateDirectory: options.stateDirectory,
      dataDirectory: options.profile.dataDirectory, version: CLI_VERSION, architecture: yield* architectureOf(process.arch),
      operation: "Install", continuation: { _tag: "Foreground", arguments: process.argv.slice(2) } }).pipe(Effect.provideServiceEffect(PreparedUpdateStore, store))
  }))
  return IdleInstallationSystem.of({
    canInstallUnattended: process.platform === "linux"
      ? executor.exitCode(Command.make("/usr/bin/sudo", "-n", "-l", "--", linuxInstalledCli, "_install-application-update",
        join(options.profile.dataDirectory, "updates", "update.json"), "--parent-stdin")).pipe(Effect.map(code => code === 0), Effect.orElseSucceed(() => false))
      : fs.access(dirname(dirname(dirname(options.runtime.resourcesDirectory))), { writable: true }).pipe(Effect.as(true), Effect.orElseSucceed(() => false)),
    isIdle: isIdle(options.profile.endpoint),
    // macOS exclusion is the installation lease the installer itself acquires.
    // This process holds its own shared lease on the lock for as long as it serves, so ask whether
    // anyone else holds or awaits it.
    installationLockFree: process.platform !== "linux" ? Effect.succeed(true)
      : linuxInstallationLockHeldByOthers(linuxInstallationLock).pipe(Effect.map(held => !held), Effect.provideService(FileSystem.FileSystem, fs),
        Effect.orElseSucceed(() => false)),
    store,
    installPrepared: Effect.gen(function* () {
      if (process.platform === "darwin") return yield* installMac
      return yield* installLinux
    }).pipe(Effect.provide([nativeHostLayer(options.addon), privateFiles])),
    notify: options.notify,
  })
})).pipe(Layer.provide(BunContext.layer))
