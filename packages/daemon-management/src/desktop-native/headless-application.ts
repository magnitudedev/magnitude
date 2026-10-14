import { Deferred, Effect, Exit, Option, Ref, Schema, Stream, type Scope } from "effect"
import { dirname } from "node:path"
import { LoginStartupFailed, type OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import type { UpdateCheckReason } from "@magnitudedev/release/hosted-update"
import { unavailableApplicationUpdate, type ApplicationUpdate } from "../application-update/application-update"
import { makeHeadlessUpdateControl, reportUpdateOutcome } from "../application-update/headless-update"
import { UpdateHardwareSource, updateHardwareFromService } from "../application-update/update-hardware"
import { WindowsPipeName, nativeWindowsPrivatePipesLayer } from "@magnitudedev/utils/windows-native"
import { NativeHost } from "./index"
import { acquireApplicationOwner } from "./application-owner"
import { serveApplicationControl, type ApplicationControlOptions } from "./application-control"
import { serveWindowsApplicationControl } from "./windows-control"
import { applicationNativeHostPath, makeApplicationService, type ApplicationRuntime, type ApplicationProfile } from "./application-bootstrap"
import type { OwnerAgent } from "./owned-service"
import { ownerDone, ownerResult, ownerUnsupported } from "./owner-agent"
import { acquireLinuxInstallationLease } from "./linux-installation-lease"
import { isUpdateInstallationActive } from "./update-installation-lease"
import { acquireMacApplicationInstallationLease, nativeMacUpdateAdmission } from "./mac-update-lease"
import { MacApplicationInstallation, NativeMacApplicationInstallation } from "./mac-update-installation"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"

export class HeadlessApplicationFailed extends Schema.TaggedError<HeadlessApplicationFailed>()("HeadlessApplicationFailed", { message: Schema.String }) {}

const windowsUncontainedServe = "On Windows, start `magnitude serve` with the Magnitude command, "
  + "%LOCALAPPDATA%\\Programs\\Magnitude CLI\\magnitude.exe, not the app's resources\\magnitude.exe. "
  + "Service wrappers such as NSSM and scheduled tasks must start that command too."

/**
 * Why a Headless owner ended. `Restart` asks the caller to admit a new owner with the same invocation;
 * `InstallUpdate` asks it to install the downloaded update now that the service is idle and stopped.
 */
export type HeadlessStop = "DesktopTakeover" | "Requested" | "Restart" | "InstallUpdate"

/**
 * Caller owns terminal/signal presentation; this scope owns admission, control and the exact service tree.
 * It returns why it stopped once the service tree, control endpoint, leases and ownership are released.
 */
export const runHeadlessApplication = (options: {
  readonly runtime: ApplicationRuntime; readonly profile: ApplicationProfile
  readonly stateDirectory: string; readonly home: string; readonly environment: Readonly<Record<string, string | undefined>>
  readonly stop: Effect.Effect<void>
  readonly stopping?: (reason: HeadlessStop) => Effect.Effect<void>
  readonly observe: (state: OwnedServiceState) => Effect.Effect<void>
  readonly initializeUpdates?: Effect.Effect<ApplicationUpdate, never, Scope.Scope | UpdateHardwareSource>
  readonly prepareStartup?: Effect.Effect<void, { readonly message: string }, Scope.Scope>
  readonly updateReady?: (version: string) => Effect.Effect<void>
  /**
   * Called while serving with each downloaded update version. It waits for an idle point and answers
   * whether to stop the service and install it; interruption means the version changed or the owner stops.
   * `report` sends a recorded outcome now.
   */
  readonly installWhenIdle?: (version: string, report: Effect.Effect<void>) => Effect.Effect<boolean>
  /**
   * Runs when the service fails before its first Ready, before the owner fails. A service that never
   * becomes ready makes no scheduled check, so `report` sends a recorded outcome before it exits.
   * Answering true stops with `InstallUpdate` instead of failing, to install a download that replaces it.
   */
  readonly startupFailed?: (report: Effect.Effect<void>, updates: ApplicationUpdate) => Effect.Effect<boolean>
}) => Effect.scoped(Effect.gen(function* () {
  const stop = yield* Deferred.make<HeadlessStop>()
  yield* options.stop.pipe(Effect.zipRight(Deferred.succeed(stop, "Requested")), Effect.forkScoped)
  const native = yield* NativeHost
  const addon = applicationNativeHostPath(options.runtime, process.platform, process.arch)
  // Only the Magnitude command launcher contains a Windows server's process tree.
  if (process.platform === "win32") yield* native.guardParent(0).pipe(Effect.mapError(() => new HeadlessApplicationFailed({ message: windowsUncontainedServe })))
  const owner = yield* acquireApplicationOwner(options.stateDirectory, { _tag: "Headless" }, options.profile.sharedControl)
  if (owner._tag !== "Owner") return yield* new HeadlessApplicationFailed({ message: "Magnitude is already running." })
  if (yield* isUpdateInstallationActive(options.stateDirectory)) return yield* new HeadlessApplicationFailed({ message: "A Magnitude update is being installed. Run `magnitude serve` when it finishes." })
  if (yield* Deferred.isDone(stop)) return yield* Deferred.await(stop)
  if (options.prepareStartup) yield* Effect.raceFirst(options.prepareStartup, Deferred.await(stop))
  if (yield* Deferred.isDone(stop)) return yield* Deferred.await(stop)
  if (options.runtime._tag === "Installed") {
    if (process.platform === "linux") yield* acquireLinuxInstallationLease(addon)
    if (process.platform === "darwin") {
      const bundle = dirname(dirname(options.runtime.resourcesDirectory))
      const active = yield* Effect.flatMap(MacApplicationInstallation, installation => installation.isInstalling(bundle)).pipe(Effect.provide(NativeMacApplicationInstallation))
      if (active) return yield* new HeadlessApplicationFailed({ message: "A Magnitude update is being installed. Run `magnitude serve` when it finishes." })
      yield* acquireMacApplicationInstallationLease(bundle).pipe(Effect.provide(nativeMacUpdateAdmission(addon)))
    }
  }
  if (yield* Deferred.isDone(stop)) return yield* Deferred.await(stop)
  const serviceState = yield* Ref.make<Option.Option<OwnedServiceState>>(Option.none())
  const updates = yield* (options.initializeUpdates ?? Effect.succeed(unavailableApplicationUpdate("Application updates require an installed Magnitude application."))).pipe(
    Effect.provideService(UpdateHardwareSource, { current: Ref.get(serviceState).pipe(Effect.map(Option.flatMap(updateHardwareFromService))) }))
  if (yield* Deferred.isDone(stop)) return yield* Deferred.await(stop)
  // Update checks start once the service is ready, so a new version reports `applied` only after it starts.
  const firstReady = yield* Deferred.make<void>()
  const report = (reason: UpdateCheckReason) => reportUpdateOutcome(updates, reason)
  const update = yield* makeHeadlessUpdateControl(updates, Deferred.await(firstReady))
  if (options.updateReady) {
    const notify = options.updateReady
    yield* updates.changes.pipe(Stream.map(state => state.transfer), Stream.changesWith((a, b) =>
      a._tag === "Ready" && b._tag === "Ready" && a.version === b.version),
      Stream.runForEach(state => state._tag === "Ready" ? notify(state.version) : Effect.void), Effect.forkScoped)
  }
  const ownerAgent: OwnerAgent = {
    state: updates.changes.pipe(Stream.map(state => ({ owner: "Headless" as const, capabilities: ["Updates" as const, "RestartService" as const, "Quit" as const], updates: Option.some(state), loginStartup: Option.none() }))),
    handle: request => {
      switch (request._tag) {
        case "CheckUpdate": return ownerResult(update("check"))
        case "DownloadUpdate": return ownerResult(update("download"))
        case "DiscardUpdate": return ownerResult(update("discard"))
        case "InstallUpdate": return ownerResult(update("install"))
        case "SetAutoDownload": return ownerResult(updates.setAutoDownload(request.enabled))
        case "Quit": return Effect.succeed(ownerDone(Deferred.succeed(stop, "Requested").pipe(Effect.asVoid)))
        // Restarting re-admits the owner, so settings that apply at service start take effect.
        case "RestartService": return Effect.succeed(ownerDone(Deferred.succeed(stop, "Restart").pipe(Effect.asVoid)))
        case "SetLoginStartup": return ownerUnsupported
      }
    },
  }
  const service = yield* makeApplicationService({ ...options, output: "Foreground", admission: "Immediate", owner: ownerAgent })
  yield* service.changes.pipe(Stream.runForEach(state => Ref.set(serviceState, Option.some(state))), Effect.forkScoped)
  const control: ApplicationControlOptions = {
    snapshot: service.state.pipe(Effect.map(state => ({ version: 1 as const, pid: process.pid, endpoint: options.profile.endpoint, owner: { _tag: "Headless" as const }, service: state }))),
    dispatch: intent => intent === "Yield" || intent === "Quit" ? Deferred.succeed(stop, intent === "Yield" ? "DesktopTakeover" : "Requested").pipe(Effect.asVoid) : Effect.void,
    login: () => new LoginStartupFailed({ message: "Login startup belongs to the desktop app. Open Magnitude to change it." }),
    update,
  }
  if (process.platform === "win32") {
    const name = yield* Schema.decodeUnknown(WindowsPipeName)(owner.socketPath)
    yield* serveWindowsApplicationControl(name, control).pipe(Effect.provide(nativeWindowsPrivatePipesLayer(addon)))
  } else yield* serveApplicationControl(owner.socketPath, control, options.profile.sharedControl)
  yield* service.changes.pipe(Stream.runForEach(state => (state._tag === "Ready" ? Deferred.succeed(firstReady, undefined) : Effect.void).pipe(
    Effect.zipRight(options.observe(state)))), Effect.forkScoped)
  if (options.installWhenIdle) {
    const decide = options.installWhenIdle
    // A newer download interrupts waiting for the previous one.
    yield* updates.changes.pipe(Stream.map(state => state.transfer), Stream.filter(transfer => transfer._tag === "Ready"),
      Stream.map(transfer => transfer.version), Stream.changes,
      Stream.flatMap(version => Stream.fromEffect(Deferred.await(firstReady).pipe(
        Effect.zipRight(decide(version, report("scheduled"))))), { switch: true }),
      Stream.filter(install => install), Stream.take(1),
      Stream.runForEach(() => Deferred.succeed(stop, "InstallUpdate")), Effect.forkScoped)
  }
  const failure = service.changes.pipe(Stream.filter(state => state._tag === "Failed" || state._tag === "CleanupFailed"), Stream.take(1), Stream.runHead,
    Effect.flatMap(state => Option.isSome(state) ? Effect.fail(new HeadlessApplicationFailed({ message: state.value.message })) : Effect.never))
  const result = yield* Effect.raceFirst(Deferred.await(stop), failure).pipe(Effect.exit)
  if (Exit.isSuccess(result) && options.stopping) yield* options.stopping(result.value)
  // An update that installed but never became ready is reported before the owner exits for a retry.
  const replace = Exit.isFailure(result) && !(yield* Deferred.isDone(firstReady)) && options.startupFailed
    ? yield* options.startupFailed(report("launch"), updates) : false
  if (replace && options.stopping) yield* options.stopping("InstallUpdate")
  yield* service.shutdown
  if (replace) return "InstallUpdate" as const
  if (Exit.isFailure(result)) return yield* Effect.failCause(result.cause)
  return result.value
})).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))
