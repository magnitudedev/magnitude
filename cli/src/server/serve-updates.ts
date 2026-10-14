import { Command, CommandExecutor, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Clock, Context, Duration, Effect, Layer, Option, Ref, Schema, Stream } from "effect"
import { dirname, join } from "node:path"
import { release } from "node:os"
import type { UpdateOwner } from "@magnitudedev/release/hosted-update"
import { type ApplicationRuntime, type ApplicationProfile, PreparedUpdateStore, UpdatePreferences, type UnattendedHold, makeUnixProcessContinuation, acquireUpdateInstallationLease,
  acquireApplicationMaintenance, nativeHostLayer, unixPrivateFilePermissions, windowsPrivateFilePermissions, recoverWindowsUpdateDirectory, nodeTerminalCommand, linuxInstallationLockHeldByOthers } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateSource, makeInstalledUpdatePreparation, makeApplicationUpdate,
  reconcilePreparedUpdate, unavailableApplicationUpdate, completeLinuxForegroundUpdate, prepareMacForegroundStartup,
  startMacForegroundInstallation, recordStartupFailure, type ApplicationUpdate } from "@magnitudedev/daemon-management/application-update"
import { CLI_VERSION } from "../version"
import { withLocalService } from "./server-reach"

const linuxInstalledCli = "/usr/lib/magnitude-desktop/resources/magnitude"
const linuxInstallationLock = "/var/lib/magnitude-desktop/installation.lock"
/** How long a deferred installation waits before trying again at the next idle point. */
const deferredRetry = Duration.minutes(15)
/** How long a failed unattended installation holds back the next unattended attempt. */
const failureBackoff = Duration.hours(1)
const idlePoll = Duration.seconds(10)
/** How long a version that cannot start waits for a newer download before exiting for its next restart. */
const replacementDownload = Duration.minutes(30)

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

/** The one line a server that cannot install by itself prints for each downloaded version. */
export const makeDownloadedNotice = (notify: (line: string) => Effect.Effect<void>) => Effect.gen(function* () {
  const noticed = yield* Ref.make(new Set<string>())
  return (version: string) => Ref.modify(noticed, seen => [!seen.has(version), new Set([...seen, version])] as const).pipe(
    Effect.flatMap(first => first ? notify(`Magnitude ${version} is downloaded. Install it with \`magnitude update install\`.`) : Effect.void))
})

/** The machine an idle installation acts on; replaced by fakes to exercise the decisions. */
export interface IdleInstallationSystem {
  /**
   * Whether this server may install without asking anyone: on Linux, sudo admits the installer
   * without a password (the service's sudoers rule, or the person's own); on macOS, a writable app folder.
   */
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
 * Install when idle: a server, the system service or `serve` run by hand, installs a downloaded
 * update only between turns whenever it can without a password. Otherwise it keeps serving and says
 * how to install; a held installation lock defers. A failed attempt, or an installed version that
 * never starts, holds back unattended installation for `failureBackoff`, and the version that failed
 * is not downloaded or installed unattended again (see `UnattendedHold`), so restarts never loop.
 */
export const makeIdleInstallation = Effect.gen(function* () {
  const system = yield* IdleInstallationSystem
  const notice = yield* makeDownloadedNotice(system.notify)

  const holdFor = (duration: Duration.Duration, failed: Option.Option<Omit<UnattendedHold, "until">>) => Effect.gen(function* () {
    const store = yield* system.store
    const previous: Option.Option<Omit<UnattendedHold, "until">> = yield* store.hold
    const held = Option.getOrElse(Option.orElse(failed, () => previous), () => ({ version: Option.none(), reason: Option.none() }))
    yield* store.recordHold({ version: held.version, reason: held.reason, until: (yield* Clock.currentTimeMillis) + Duration.toMillis(duration) })
  })

  const defer = (version: string) => Effect.gen(function* () {
    const store = yield* system.store
    const from = yield* store.read.pipe(Effect.map(Option.flatMap(pending => pending.from)), Effect.orElseSucceed(() => Option.none()))
    yield* store.recordOutcome({ outcome: "deferred", version, from, reason: Option.none() })
    yield* holdFor(deferredRetry, Option.none())
    yield* system.notify(`Magnitude ${version} is waiting for another Magnitude to quit; trying again later.`)
  })

  const holdUntil = system.store.pipe(Effect.flatMap(store => store.hold),
    Effect.map(hold => Option.match(hold, { onNone: () => 0, onSome: hold => hold.until })), Effect.orElseSucceed(() => 0))

  /**
   * Resolves true at an idle point where the installation lock is free, so the caller may stop and
   * install. The package refuses to upgrade while anyone else holds that lock, so a held lock defers
   * without stopping: the deferral is reported through `report` and the current version keeps serving.
   */
  const installWhenIdle = (version: string, report: Effect.Effect<void> = Effect.void) => Effect.gen(function* () {
    if (!(yield* system.canInstallUnattended)) {
      yield* notice(version)
      return false
    }
    while (true) {
      const wait = (yield* holdUntil) - (yield* Clock.currentTimeMillis)
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
    // A failed package keeps only its outcome and stays held back, so the next check can fetch a
    // newer version without fetching this one again. Any other failure keeps the download for later.
    const store = yield* system.store
    const pending = yield* store.read
    const failed = Option.filter(pending, update => update.installation._tag === "Failed")
    yield* holdFor(failureBackoff, Option.map(failed, update => ({ version: Option.some(update.release.version),
      reason: Option.some(update.installation._tag === "Failed" ? update.installation.kind : "install" as const) })))
    if (Option.isSome(failed) && failed.value.installation._tag === "Failed") {
      yield* store.recordOutcome({ outcome: "failed", version: failed.value.release.version, from: failed.value.from, reason: Option.some(failed.value.installation.kind) })
      yield* store.discard
    }
  }).pipe(Effect.ignore, Effect.zipRight(system.notify(`The update could not be installed: ${error.message} Serving the current version.`)))))

  /**
   * An applied version that never became ready is recorded as `failed`/`startup`, held back and
   * reported once. While the running version is the held one, restarts still check for updates, at
   * most once per `failureBackoff`, and resolve true once a newer download is ready, so the caller
   * installs it in place of the version that cannot start.
   */
  const startupFailed = (report: Effect.Effect<void>, updates: ApplicationUpdate) => system.store.pipe(Effect.flatMap(store => Effect.gen(function* () {
    const first = yield* recordStartupFailure(CLI_VERSION, failureBackoff).pipe(Effect.provideService(PreparedUpdateStore, store))
    const held = Option.filter(yield* store.hold, hold => Option.exists(hold.version, version => version === CLI_VERSION))
    if (Option.isNone(held)) {
      if (Option.isSome(yield* store.outcome)) yield* report
      return false
    }
    const now = yield* Clock.currentTimeMillis
    if (!first) {
      if (held.value.until > now) return false
      yield* store.recordHold({ ...held.value, until: now + Duration.toMillis(failureBackoff) })
    }
    yield* report
    if (!(yield* system.canInstallUnattended)) return false
    // An offer that downloads automatically is about to start downloading, so it is not settled yet.
    const settled = yield* updates.changes.pipe(Stream.filter(state => state.transfer._tag === "Available"
      ? Option.isSome(state.transfer.failed) || !(state.preference._tag === "Known" && state.preference.autoDownload)
      : state.transfer._tag !== "Downloading" && state.transfer._tag !== "Staging" && state.transfer._tag !== "Cancelling"),
      Stream.map(state => state.transfer),
      Stream.runHead, Effect.timeout(replacementDownload))
    return Option.exists(settled, transfer => transfer._tag === "Ready")
  })), Effect.orElseSucceed(() => false))

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
    yield* acquireApplicationMaintenance(options.stateDirectory, options.profile.sharedControl)
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
