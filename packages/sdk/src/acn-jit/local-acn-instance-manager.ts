import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import {
  AcnHealthResponseSchema,
  AcnReady,
  type AcnHealthResponse,
  type AcnInstance,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"
import {
  ExactProcessController,
  ExactProcessControllerLive,
  makeAcnOwnerLock,
  makeAcnRevisionStore,
  reapProcessTree,
  retryStoreObservation,
  SqliteMutex,
  COORDINATION_POLL_INTERVAL,
  type AcnProcessStoreError,
  type AcnOwnerLock,
  type AcnOwnerRecord,
  type AcnRevisionStore,
  type ExactProcess,
  type ExactProcessController as ExactProcessControllerService,
} from "@magnitudedev/acn-protocol/coordination"
import type { ArtifactInstallationEvent } from "@magnitudedev/release"
import { Array as Arr, Duration, Effect, Option, Schema, Scope, Stream } from "effect"
import { defaultDataDir, resolveBinaryCommand, type BinaryAcquisitionEvent } from "../binary"
import {
  SDK_ACN_BUILD_KIND,
  SDK_ACN_DEVELOPMENT_KEY,
  SDK_ACN_TARGET,
} from "../version"
import {
  AcnInstanceManager,
  type AcnEnsureEvent,
} from "./acn-instance-manager"
import { ChildProcessSpawner } from "./child-process"
import {
  AcnAdministrationFailed,
  AcnEnsuranceError,
  AcnEnsuranceFailed,
  type AcnEnsuranceError as AcnEnsuranceErrorType,
} from "./errors"
import { acnLifecycleObservationFromHealthState } from "./lifecycle"

type ReadyInstance = AcnInstance<AcnReady>

export interface AcnLaunchOverride {
  readonly target: AcnTarget
  readonly command: Arr.NonEmptyReadonlyArray<string>
}

export interface LocalAcnInstanceManagerOptions {
  readonly binaryPath?: string
  readonly dataDir?: string
  readonly debug?: boolean
  readonly launchOverride?: AcnLaunchOverride
}

interface PreparedCommand {
  readonly target: AcnTarget
  readonly command: Arr.NonEmptyReadonlyArray<string>
}

interface HealthObservation {
  readonly status: number
  readonly health: AcnHealthResponse
}

const HEALTH_TIMEOUT = Duration.seconds(2)
const GRACEFUL_STOP_WAIT = Duration.seconds(5)

const inspectProcess = (
  processes: ExactProcessControllerService,
  pid: number,
): Effect.Effect<Option.Option<ExactProcess["processStartIdentity"]>> => Effect.suspend(() =>
  processes.inspect(pid).pipe(
    Effect.catchAll((error) => Effect.logWarning("ACN process inspection is temporarily unavailable").pipe(
      Effect.annotateLogs({ pid, operation: error.operation }),
      Effect.zipRight(Effect.sleep(COORDINATION_POLL_INTERVAL)),
      Effect.zipRight(inspectProcess(processes, pid)),
    )),
  ),
)

const storeEnsuranceError = (error: AcnProcessStoreError): AcnEnsuranceErrorType =>
  new AcnEnsuranceFailed({
    reason: `${error._tag} during ${"operation" in error ? error.operation : "validation"} at ${error.path}: ${error.message}`,
  })

const sameOwner = (left: AcnOwnerRecord, right: AcnOwnerRecord): boolean =>
  left.pid === right.pid &&
  left.processStartIdentity === right.processStartIdentity &&
  left.port === right.port

const exactFrom = (owner: AcnOwnerRecord): ExactProcess => ({
  pid: owner.pid,
  processStartIdentity: owner.processStartIdentity,
})

const artifactProgress = (
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>,
) => ({
  completed: event.progress.acceptedBytes,
  totalBytes: event.progress.totalBytes,
  unit: "Bytes" as const,
  attempt: Option.some(event.progress.attempt),
})

const registerTarget = (
  store: AcnRevisionStore,
  target: AcnTarget,
): Effect.Effect<void, AcnEnsuranceErrorType, Scope.Scope> =>
  SDK_ACN_BUILD_KIND === "published"
    ? retryStoreObservation(store.registerPublished(target.revision)).pipe(
        Effect.mapError(storeEnsuranceError),
      )
    : Effect.gen(function* () {
        if (SDK_ACN_DEVELOPMENT_KEY === undefined) {
          return yield* new AcnEnsuranceFailed({ reason: "Development ACN target has no development key" })
        }
        const hold = yield* retryStoreObservation(
          store.holdDevelopment(target.revision, SDK_ACN_DEVELOPMENT_KEY),
        ).pipe(Effect.mapError(storeEnsuranceError))
        // The hold is released via scope finalizer when the manager's scope closes.
        void hold
      })

const sameTarget = (left: AcnTarget, right: AcnTarget): boolean =>
  left.revision === right.revision && left.identity === right.identity

export const makeLocalAcnInstanceManager = (
  options: LocalAcnInstanceManagerOptions = {},
): Effect.Effect<
  AcnInstanceManager,
  never,
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor
  | Path.Path
  | ChildProcessSpawner
  | SqliteMutex
  | Scope.Scope
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const http = yield* HttpClient.HttpClient
  const commandExecutor = yield* CommandExecutor.CommandExecutor
  const path = yield* Path.Path
  const spawner = yield* ChildProcessSpawner
  const processes = yield* ExactProcessController
  const dataDirectory = options.dataDir ?? defaultDataDir()
  const store = yield* makeAcnRevisionStore(dataDirectory).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.provideService(Path.Path, path),
  )
  const ownerLock = yield* makeAcnOwnerLock(dataDirectory).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.provideService(Path.Path, path),
  )

  const probeHealth = (owner: AcnOwnerRecord): Effect.Effect<Option.Option<HealthObservation>, AcnEnsuranceErrorType> =>
    http.execute(HttpClientRequest.get(`http://127.0.0.1:${owner.port}/health`)).pipe(
      Effect.timeoutOption(HEALTH_TIMEOUT),
      Effect.option,
      Effect.flatMap(Option.match({
        onNone: () => Effect.succeed(Option.none()),
        onSome: Option.match({
          onNone: () => Effect.succeed(Option.none()),
          onSome: (response) => response.json.pipe(
            Effect.flatMap(Schema.decodeUnknown(AcnHealthResponseSchema)),
            Effect.map((health) => Option.some({ status: response.status, health })),
            Effect.mapError((error) => new AcnEnsuranceFailed({
              reason: `ACN owner returned malformed health: ${String(error)}`,
            })),
          ),
        }),
      })),
    )

  const resolveCommand = (
    target: AcnTarget,
    emit: (event: AcnEnsureEvent) => void,
  ): Effect.Effect<PreparedCommand, AcnEnsuranceErrorType> => {
    if (options.launchOverride !== undefined) {
      return sameTarget(options.launchOverride.target, target)
        ? Effect.succeed(options.launchOverride)
        : Effect.fail(new AcnEnsuranceFailed({
            reason: `This client cannot launch selected ACN revision ${target.revision}`,
          }))
    }
    let plan = Option.none<{
      readonly daemonBytes: number
      readonly inferenceEngineBytes: number
      readonly inferenceEngineBytesExact: boolean
    }>()
    const report = (event: BinaryAcquisitionEvent) => Effect.sync(() => {
      if (event._tag === "Planned") plan = Option.some(event.plan)
      else if (event.event._tag === "Downloading" && Option.isSome(plan)) {
        emit({
          _tag: "Observation",
          observation: {
            _tag: "Installing",
            phase: "DownloadingDaemon",
            plan: plan.value,
            progress: Option.some(artifactProgress(event.event)),
          },
        })
      }
    })
    return resolveBinaryCommand({
      binaryPath: options.binaryPath,
      version: target.identity,
      acnRevision: target.revision,
      dataDir: dataDirectory,
      acquisitionObserver: Option.some({ report }),
    }).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provideService(HttpClient.HttpClient, http),
      Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
      Effect.provideService(Path.Path, path),
      Effect.map((resolved) => ({ target, command: resolved.command })),
    )
  }

  const observeReady = (
    selected: AcnTarget["revision"],
    owner: AcnOwnerRecord,
    emit: (event: AcnEnsureEvent) => void,
  ): Effect.Effect<Option.Option<ReadyInstance>, AcnEnsuranceErrorType> =>
    Effect.gen(function* () {
      const identity = yield* inspectProcess(processes, owner.pid)
      if (!Option.contains(identity, owner.processStartIdentity)) return Option.none()
      const observed = yield* probeHealth(owner)
      if (Option.isNone(observed)) return Option.none()
      const { health, status } = observed.value
      if (health.pid !== owner.pid) {
        return yield* new AcnEnsuranceFailed({ reason: "ACN health PID contradicts locked owner metadata" })
      }
      if (health.revision > selected) {
        return yield* new AcnEnsuranceFailed({
          reason: "ACN health revision contradicts the selected coordination revision",
        })
      }
      if (health.revision < selected) return Option.none()
      if (status !== 200 && status !== 503) {
        return yield* new AcnEnsuranceFailed({ reason: `ACN health returned unexpected status ${status}` })
      }
      if ((status === 200) !== (health.state._tag === "Ready")) {
        return yield* new AcnEnsuranceFailed({ reason: "ACN health status contradicts its lifecycle state" })
      }
      const progress = acnLifecycleObservationFromHealthState(health.state)
      if (Option.isSome(progress)) emit({ _tag: "Observation", observation: progress.value })
      if (status !== 200) return Option.none()

      const confirmedRevision = yield* retryStoreObservation(store.selected).pipe(
        Effect.mapError(storeEnsuranceError),
      )
      if (!Option.contains(confirmedRevision, selected)) return Option.none()
      const confirmedOwner = yield* retryStoreObservation(ownerLock.observe).pipe(
        Effect.mapError(storeEnsuranceError),
      )
      if (confirmedOwner._tag !== "Locked" || !sameOwner(confirmedOwner.owner, owner)) return Option.none()
      const confirmedIdentity = yield* inspectProcess(processes, owner.pid)
      if (!Option.contains(confirmedIdentity, owner.processStartIdentity)) return Option.none()
      return Option.some({
        revision: health.revision,
        id: health.id,
        identity: health.version,
        url: `http://127.0.0.1:${owner.port}`,
        pid: owner.pid,
        processStartIdentity: owner.processStartIdentity,
        lifecycle: new AcnReady({}),
      })
    })

  const ensureEffect = (
    target: AcnTarget,
    emit: (event: AcnEnsureEvent) => void,
  ): Effect.Effect<ReadyInstance, AcnEnsuranceErrorType, Scope.Scope> =>
    Effect.gen(function* () {
      let prepared = Option.none<PreparedCommand>()
      let launched = Option.none<ExactProcess>()
      while (true) {
        const selected = yield* retryStoreObservation(store.selected).pipe(
          Effect.mapError(storeEnsuranceError),
        )
        const owner = yield* retryStoreObservation(ownerLock.observe).pipe(
          Effect.mapError(storeEnsuranceError),
        )
        if (owner._tag === "Locked") {
          if (Option.isSome(selected) && selected.value >= target.revision) {
            const ready = yield* observeReady(selected.value, owner.owner, emit)
            if (Option.isSome(ready)) return ready.value
          }
          if (Option.isNone(prepared) &&
            (Option.isNone(selected) || selected.value < target.revision)) {
            if (!sameTarget(target, SDK_ACN_TARGET)) {
              return yield* new AcnEnsuranceFailed({
                reason: `Selected ACN revision cannot be advanced to ${target.revision} by this client`,
              })
            }
            prepared = Option.some(yield* resolveCommand(target, emit))
            yield* registerTarget(store, target)
          }
          yield* Effect.sleep(COORDINATION_POLL_INTERVAL)
          continue
        }
        if (owner._tag === "Publishing") {
          yield* Effect.sleep(COORDINATION_POLL_INTERVAL)
          continue
        }
        if (Option.isSome(selected) && selected.value > target.revision) {
          return yield* new AcnEnsuranceFailed({
            reason: `Selected ACN revision ${selected.value} cannot be launched by this client`,
          })
        }
        if (Option.isNone(prepared)) {
          if (Option.isNone(selected) || selected.value < target.revision) {
            if (!sameTarget(target, SDK_ACN_TARGET)) {
              return yield* new AcnEnsuranceFailed({
                reason: `Selected ACN revision cannot be advanced to ${target.revision} by this client`,
              })
            }
            prepared = Option.some(yield* resolveCommand(target, emit))
            yield* registerTarget(store, target)
            continue
          }
          prepared = Option.some(yield* resolveCommand(target, emit))
        }
        if (Option.isNone(selected) || selected.value !== target.revision) {
          yield* Effect.sleep(COORDINATION_POLL_INTERVAL)
          continue
        }
        if (Option.isNone(prepared)) {
          return yield* new AcnEnsuranceFailed({ reason: "Selected ACN launch material was not prepared" })
        }
        const command = prepared.value.command
        if (Option.isSome(launched)) {
          const alive = yield* inspectProcess(processes, launched.value.pid)
          if (Option.contains(alive, launched.value.processStartIdentity)) {
            yield* Effect.sleep(COORDINATION_POLL_INTERVAL)
            continue
          }
          launched = Option.none()
        }
        const argv = [
          ...command,
          ...(options.debug === true && !command.includes("--debug") ? ["--debug"] : []),
          "--wait-for-handoff",
          "--data-dir",
          dataDirectory,
        ]
        if (!Arr.isNonEmptyReadonlyArray(argv)) {
          return yield* new AcnEnsuranceFailed({ reason: "Cannot spawn an empty ACN command" })
        }
        const child = yield* spawner.spawn(argv)
        const identity = yield* inspectProcess(processes, child.pid).pipe(
          Effect.flatMap(Option.match({
            onNone: () => Effect.fail(new AcnEnsuranceFailed({
              reason: `Spawned ACN ${child.pid} exited before handoff`,
            })),
            onSome: Effect.succeed,
          })),
        )
        launched = Option.some({ pid: child.pid, processStartIdentity: identity })
        yield* child.handoff
      }
    })

  const ensure: AcnInstanceManager["ensure"] = (request) =>
    Stream.asyncPush<AcnEnsureEvent, AcnEnsuranceErrorType>((sink) => {
      return Effect.forkScoped(ensureEffect(request.target, (event) => sink.single(event)).pipe(
        Effect.match({
          onFailure: sink.fail,
          onSuccess: (instance) => {
            sink.single({ _tag: "Ready", instance })
            sink.end()
          },
        }),
      ))
    }, { bufferSize: "unbounded" })

  const stop = Effect.gen(function* () {
    const observation = yield* ownerLock.observe
    if (observation._tag !== "Locked") return
    const owner = observation.owner
    const exact = exactFrom(owner)
    const identity = yield* processes.inspect(owner.pid)
    if (!Option.contains(identity, owner.processStartIdentity)) return
    yield* http.execute(HttpClientRequest.post(`http://127.0.0.1:${owner.port}/shutdown`)).pipe(
      Effect.timeout(GRACEFUL_STOP_WAIT),
      Effect.ignore,
    )
    if (yield* reapProcessTree(processes, exact)) return
    return yield* new AcnAdministrationFailed({
      reason: `Could not prove ACN process tree ${owner.pid} absent`,
    })
  }).pipe(Effect.mapError((error) => error instanceof AcnAdministrationFailed
    ? error
    : new AcnAdministrationFailed({ reason: String(error) })))

  return AcnInstanceManager.of({ ensure, stop })
}).pipe(Effect.provideService(ExactProcessController, ExactProcessControllerLive))
