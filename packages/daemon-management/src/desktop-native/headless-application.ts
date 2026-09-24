import { Deferred, Effect, Exit, Option, Schema, Stream, type Scope } from "effect"
import { dirname } from "node:path"
import { LoginStartupFailed, type OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import { unavailableApplicationUpdate, type ApplicationUpdate } from "../application-update/application-update"
import { makeHeadlessUpdateControl } from "../application-update/headless-update"
import { WindowsPipeName, nativeWindowsPrivatePipesLayer } from "@magnitudedev/utils/windows-native"
import { NativeHost } from "./index"
import { acquireApplicationOwner } from "./application-owner"
import { serveApplicationControl, type ApplicationControlOptions } from "./application-control"
import { serveWindowsApplicationControl } from "./windows-control"
import { applicationNativeHostPath, makeApplicationService, type ApplicationRuntime, type ApplicationProfile } from "./application-bootstrap"
import { acquireLinuxInstallationLease } from "./linux-installation-lease"
import { isUpdateInstallationActive } from "./update-installation-lease"
import { acquireMacApplicationInstallationLease, nativeMacUpdateAdmission } from "./mac-update-lease"
import { MacApplicationInstallation, NativeMacApplicationInstallation } from "./mac-update-installation"
import { ProcessGroupController } from "@magnitudedev/utils/process-groups"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"

export class HeadlessApplicationFailed extends Schema.TaggedError<HeadlessApplicationFailed>()("HeadlessApplicationFailed", { message: Schema.String }) {}

/** Caller owns terminal/signal presentation; this scope owns admission, control and the exact service tree. */
export const runHeadlessApplication = (options: {
  readonly runtime: ApplicationRuntime; readonly profile: ApplicationProfile
  readonly stateDirectory: string; readonly home: string; readonly environment: Readonly<Record<string, string | undefined>>
  readonly stop: Effect.Effect<void>
  readonly observe: (state: OwnedServiceState) => Effect.Effect<void>
  readonly initializeUpdates?: Effect.Effect<ApplicationUpdate, never, Scope.Scope>
  readonly prepareStartup?: Effect.Effect<void, { readonly message: string }, Scope.Scope>
  readonly updateReady?: (version: string) => Effect.Effect<void>
}) => Effect.scoped(Effect.gen(function* () {
  const stop = yield* Deferred.make<void>()
  yield* options.stop.pipe(Effect.zipRight(Deferred.succeed(stop, undefined)), Effect.forkScoped)
  const native = yield* NativeHost
  const addon = applicationNativeHostPath(options.runtime, process.platform, process.arch)
  if (process.platform === "win32") yield* native.guardParent(0)
  const owner = yield* acquireApplicationOwner(options.stateDirectory, { _tag: "Headless" })
  if (owner._tag !== "Owner") return yield* new HeadlessApplicationFailed({ message: "Magnitude is already running." })
  if (yield* isUpdateInstallationActive(options.stateDirectory)) return yield* new HeadlessApplicationFailed({ message: "A Magnitude update is being installed. Run `magnitude serve` when it finishes." })
  if (yield* Deferred.isDone(stop)) return
  if (options.prepareStartup) yield* Effect.raceFirst(options.prepareStartup, Deferred.await(stop))
  if (yield* Deferred.isDone(stop)) return
  if (options.runtime._tag === "Installed") {
    if (process.platform === "linux") yield* acquireLinuxInstallationLease(addon)
    if (process.platform === "darwin") {
      const bundle = dirname(dirname(options.runtime.resourcesDirectory))
      const active = yield* Effect.flatMap(MacApplicationInstallation, installation => installation.isInstalling(bundle)).pipe(Effect.provide(NativeMacApplicationInstallation))
      if (active) return yield* new HeadlessApplicationFailed({ message: "A Magnitude update is being installed. Run `magnitude serve` when it finishes." })
      yield* acquireMacApplicationInstallationLease(bundle).pipe(Effect.provide(nativeMacUpdateAdmission(addon)))
    }
  }
  if (yield* Deferred.isDone(stop)) return
  const updates = yield* options.initializeUpdates ?? Effect.succeed(unavailableApplicationUpdate("Application updates require an installed Magnitude application."))
  if (yield* Deferred.isDone(stop)) return
  const update = yield* makeHeadlessUpdateControl(updates)
  if (options.updateReady) {
    const notify = options.updateReady
    yield* updates.changes.pipe(Stream.map(state => state.transfer), Stream.changesWith((a, b) =>
      a._tag === "Ready" && b._tag === "Ready" && a.version === b.version),
      Stream.runForEach(state => state._tag === "Ready" ? notify(state.version) : Effect.void), Effect.forkScoped)
  }
  const service = yield* makeApplicationService({ ...options, output: "Foreground", admission: "Immediate" })
  const control: ApplicationControlOptions = {
    snapshot: service.state.pipe(Effect.map(state => ({ version: 1 as const, pid: process.pid, endpoint: options.profile.endpoint, owner: { _tag: "Headless" as const }, service: state }))),
    dispatch: intent => intent === "Yield" || intent === "Quit" ? Deferred.succeed(stop, undefined).pipe(Effect.asVoid) : Effect.void,
    login: () => new LoginStartupFailed({ message: "Login startup belongs to the desktop app. Open Magnitude to change it." }),
    update,
  }
  if (process.platform === "win32") {
    const name = yield* Schema.decodeUnknown(WindowsPipeName)(owner.socketPath)
    yield* serveWindowsApplicationControl(name, control).pipe(Effect.provide(nativeWindowsPrivatePipesLayer(addon)))
  } else yield* serveApplicationControl(owner.socketPath, control)
  yield* service.changes.pipe(Stream.runForEach(options.observe), Effect.forkScoped)
  const failure = service.changes.pipe(Stream.filter(state => state._tag === "Failed" || state._tag === "CleanupFailed"), Stream.take(1), Stream.runHead,
    Effect.flatMap(state => Option.isSome(state) ? Effect.fail(new HeadlessApplicationFailed({ message: state.value.message })) : Effect.never))
  const result = yield* Effect.raceFirst(Deferred.await(stop), failure).pipe(Effect.exit)
  yield* service.shutdown
  if (Exit.isFailure(result)) return yield* Effect.failCause(result.cause)
})).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))
