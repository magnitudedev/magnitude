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
  makeAcnOwnerStore,
  SqliteDriver,
  COORDINATION_POLL_INTERVAL,
  TREE_KILL_WAIT,
  TREE_TERM_WAIT,
  waitForTreeAbsence,
  type AcnOwnerRecord,
  type AcnProcessStoreError,
  type ExactProcess,
  type ExactProcessController as ExactProcessControllerService,
} from "@magnitudedev/acn-protocol/coordination"
import type { ArtifactInstallationEvent } from "@magnitudedev/release"
import {
  Array as Arr,
  Clock,
  Duration,
  Effect,
  Match,
  Option,
  Schedule,
  Schema,
  Scope,
  Stream,
} from "effect"
import { defaultDataDir, resolveBinaryCommand, type BinaryAcquisitionEvent } from "../binary"
import { SDK_ACN_TARGET } from "../version"
import {
  ACN_ENSURE_TIMEOUT,
  AcnInstanceManager,
  type AcnEnsureEvent,
} from "./acn-instance-manager"
import { ChildProcessSpawner, type SpawnedAcnCandidate } from "./child-process"
import {
  AcnAdministrationFailed,
  AcnEnsuranceFailed,
  type AcnEnsuranceError as AcnEnsuranceErrorType,
} from "./errors"
import {
  acnLifecycleObservationFromHealthState,
} from "./lifecycle"

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

interface LaunchedCandidate {
  readonly process: ExactProcess
  readonly child: SpawnedAcnCandidate
  readonly launchedAt: number
}

type ConvergenceState =
  | { readonly _tag: "SurvivingPredecessorTree"; readonly owner: AcnOwnerRecord }
  | {
      readonly _tag: "ObservableOwner"
      readonly owner: AcnOwnerRecord
      readonly observed: Option.Option<HealthObservation>
      readonly now: number
    }
  | { readonly _tag: "CandidatePending" }
  | {
      readonly _tag: "CandidateExited"
      readonly candidate: LaunchedCandidate
      readonly code: number
      readonly stderr: string
    }
  | { readonly _tag: "CandidateAdmissionExpired"; readonly candidate: LaunchedCandidate }
  | { readonly _tag: "LaunchCandidate" }
  | { readonly _tag: "LaunchOccurrenceLost" }

const HEALTH_TIMEOUT = Duration.seconds(2)
const HEALTH_GRACE = Duration.seconds(30)
const STARTUP_CEILING = Duration.minutes(5)
const STOPPING_GRACE = Duration.seconds(5)
const CANDIDATE_ADMISSION_TIMEOUT = Duration.seconds(30)
const CANDIDATE_PARENT_RELEASE_TIMEOUT = Duration.seconds(2)
const GRACEFUL_STOP_WAIT = Duration.seconds(5)
const STORE_RETRY_INTERVAL = Duration.millis(25)
const STORE_OPERATION_TIMEOUT = Duration.seconds(30)
const PROCESS_OPERATION_TIMEOUT = Duration.seconds(30)

const monotonicMillis = Clock.currentTimeNanos.pipe(
  Effect.map((nanos) => Number(nanos / 1_000_000n)),
)

const sameOwner = (left: AcnOwnerRecord, right: AcnOwnerRecord): boolean =>
  left.pid === right.pid &&
  left.processStartIdentity === right.processStartIdentity &&
  left.port === right.port

const ownerNamesProcess = (owner: AcnOwnerRecord, process: ExactProcess): boolean =>
  owner.pid === process.pid && owner.processStartIdentity === process.processStartIdentity

const ownerKey = (owner: AcnOwnerRecord): string =>
  `${owner.pid}:${owner.processStartIdentity}:${owner.port}`

const exactFrom = (owner: AcnOwnerRecord): ExactProcess => ({
  pid: owner.pid,
  processStartIdentity: owner.processStartIdentity,
})

const storeFailure = (error: AcnProcessStoreError): AcnEnsuranceFailed =>
  new AcnEnsuranceFailed({
    reason: `${error._tag} during ${"operation" in error ? error.operation : "validation"} at ${error.path}${"message" in error ? `: ${error.message}` : ""}`,
  })

const retryStore = <A>(
  effect: Effect.Effect<A, AcnProcessStoreError>,
): Effect.Effect<A, AcnEnsuranceFailed> => effect.pipe(
  Effect.retry({
    schedule: Schedule.spaced(STORE_RETRY_INTERVAL),
    while: (error) => error._tag !== "AcnProcessStoreInvalid",
  }),
  Effect.timeoutFail({
    duration: STORE_OPERATION_TIMEOUT,
    onTimeout: () => new AcnEnsuranceFailed({ reason: "ACN coordination store remained unavailable" }),
  }),
  Effect.mapError((error) => error instanceof AcnEnsuranceFailed ? error : storeFailure(error)),
)

const inspectProcess = (
  processes: ExactProcessControllerService,
  pid: number,
): Effect.Effect<Option.Option<ExactProcess["processStartIdentity"]>, AcnEnsuranceFailed> =>
  processes.inspect(pid).pipe(
    Effect.retry(Schedule.spaced(COORDINATION_POLL_INTERVAL)),
    Effect.timeoutFail({
      duration: PROCESS_OPERATION_TIMEOUT,
      onTimeout: () => new AcnEnsuranceFailed({
        reason: `Exact process inspection remained unavailable for PID ${pid}`,
      }),
    }),
    Effect.mapError((error) => error instanceof AcnEnsuranceFailed
      ? error
      : new AcnEnsuranceFailed({ reason: error.message })),
  )

const artifactProgress = (
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>,
) => ({
  completed: event.progress.acceptedBytes,
  totalBytes: event.progress.totalBytes,
  unit: "Bytes" as const,
  attempt: Option.some(event.progress.attempt),
})

const sameTarget = (left: AcnTarget, right: AcnTarget): boolean =>
  left.revision === right.revision && left.identity === right.identity

/** @internal Effect-DI constructor used by the live wrapper and deterministic tests. */
export const makeLocalAcnInstanceManagerWithProcessController = (
  options: LocalAcnInstanceManagerOptions = {},
): Effect.Effect<
  AcnInstanceManager,
  never,
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor
  | Path.Path
  | ChildProcessSpawner
  | SqliteDriver
  | ExactProcessController
  | Scope.Scope
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const http = yield* HttpClient.HttpClient
  const commandExecutor = yield* CommandExecutor.CommandExecutor
  const path = yield* Path.Path
  const spawner = yield* ChildProcessSpawner
  const processes = yield* ExactProcessController
  const dataDirectory = options.dataDir ?? defaultDataDir()
  const owners = yield* makeAcnOwnerStore(dataDirectory).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.provideService(Path.Path, path),
  )

  const probeHealth = (
    owner: AcnOwnerRecord,
  ): Effect.Effect<Option.Option<HealthObservation>> =>
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
            Effect.catchAll(() => Effect.succeed(Option.none())),
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
            reason: `This client cannot launch ACN revision ${target.revision}`,
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

  const ownerStillCurrent = (owner: AcnOwnerRecord): Effect.Effect<boolean, AcnEnsuranceFailed> =>
    retryStore(owners.current).pipe(
      Effect.map(Option.exists((current) => sameOwner(current, owner))),
    )

  const ownerStillSafeToSignal = (
    owner: AcnOwnerRecord,
  ): Effect.Effect<boolean, AcnEnsuranceFailed> => Effect.gen(function* () {
    if (!(yield* ownerStillCurrent(owner))) return false
    const identity = yield* inspectProcess(processes, owner.pid)
    if (Option.isSome(identity) && identity.value !== owner.processStartIdentity) {
      return yield* new AcnEnsuranceFailed({
        reason: `PID ${owner.pid} now identifies a different process occurrence`,
      })
    }
    return true
  })

  const retireOwner = (
    owner: AcnOwnerRecord,
  ): Effect.Effect<void, AcnEnsuranceFailed> => Effect.gen(function* () {
    const exact = exactFrom(owner)
    if (yield* processes.treeAbsent(exact).pipe(
      Effect.mapError((error) => new AcnEnsuranceFailed({ reason: error.message })),
    )) return

    const identity = yield* inspectProcess(processes, owner.pid)
    if (Option.contains(identity, owner.processStartIdentity) && (yield* ownerStillCurrent(owner))) {
      yield* http.execute(HttpClientRequest.post(`http://127.0.0.1:${owner.port}/shutdown`)).pipe(
        Effect.timeout(HEALTH_TIMEOUT),
        Effect.ignore,
      )
      if (yield* waitForTreeAbsence(processes, exact, GRACEFUL_STOP_WAIT).pipe(
        Effect.mapError((error) => new AcnEnsuranceFailed({ reason: error.message })),
      )) return
    }
    if (!(yield* ownerStillSafeToSignal(owner))) return
    yield* processes.signalTree(exact, "term").pipe(
      Effect.mapError((error) => new AcnEnsuranceFailed({ reason: error.message })),
    )
    if (yield* waitForTreeAbsence(processes, exact, TREE_TERM_WAIT).pipe(
      Effect.mapError((error) => new AcnEnsuranceFailed({ reason: error.message })),
    )) return
    if (!(yield* ownerStillSafeToSignal(owner))) return
    yield* processes.signalTree(exact, "kill").pipe(
      Effect.mapError((error) => new AcnEnsuranceFailed({ reason: error.message })),
    )
    if (!(yield* waitForTreeAbsence(processes, exact, TREE_KILL_WAIT).pipe(
      Effect.mapError((error) => new AcnEnsuranceFailed({ reason: error.message })),
    ))) {
      return yield* new AcnEnsuranceFailed({
        reason: `Could not prove ACN process tree ${owner.pid} absent`,
      })
    }
  })

  const readyInstance = (
    owner: AcnOwnerRecord,
    observed: HealthObservation,
  ): Effect.Effect<Option.Option<ReadyInstance>, AcnEnsuranceFailed> =>
    Effect.gen(function* () {
      const { health, status } = observed
      if (status !== 200 || health.state._tag !== "Ready") return Option.none()
      const confirmedOwner = yield* retryStore(owners.current)
      if (!Option.exists(confirmedOwner, (current) => sameOwner(current, owner))) return Option.none()
      const identity = yield* inspectProcess(processes, owner.pid)
      if (!Option.contains(identity, owner.processStartIdentity)) return Option.none()
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
  ): Effect.Effect<ReadyInstance, AcnEnsuranceErrorType, Scope.Scope> => {
    const run = Effect.gen(function* () {
      let prepared = Option.none<PreparedCommand>()
      let launched = Option.none<LaunchedCandidate>()
      let hasLaunched = false
      let observedOwner = ""
      let observedHealthState = ""
      let observedHealthStateSince = yield* monotonicMillis
      let ownerObservedAt = observedHealthStateSince

      const prepare = Effect.gen(function* () {
        if (Option.isSome(prepared)) return prepared.value
        if (!sameTarget(target, SDK_ACN_TARGET)) {
          return yield* new AcnEnsuranceFailed({
            reason: `This client cannot launch ACN revision ${target.revision}`,
          })
        }
        const value = yield* resolveCommand(target, emit)
        prepared = Option.some(value)
        return value
      })

      const classifyWithoutLiveOwner = (
        now: number,
      ): Effect.Effect<ConvergenceState> => Effect.gen(function* () {
        if (Option.isSome(launched)) {
          const candidate = launched.value
          const exited = yield* candidate.child.exited.pipe(Effect.timeoutOption(Duration.millis(1)))
          if (Option.isSome(exited)) {
            return { _tag: "CandidateExited", candidate, ...exited.value }
          }
          return now - candidate.launchedAt >= Duration.toMillis(CANDIDATE_ADMISSION_TIMEOUT)
            ? { _tag: "CandidateAdmissionExpired", candidate }
            : { _tag: "CandidatePending" }
        }
        return hasLaunched
          ? { _tag: "LaunchOccurrenceLost" }
          : { _tag: "LaunchCandidate" }
      })

      while (true) {
        const now = yield* monotonicMillis
        const owner = yield* retryStore(owners.current)

        const state = yield* Option.match(owner, {
          onNone: () => classifyWithoutLiveOwner(now),
          onSome: (current): Effect.Effect<ConvergenceState, AcnEnsuranceFailed> =>
            Effect.gen(function* () {
              if (Option.isSome(launched) && ownerNamesProcess(current, launched.value.process)) {
                yield* launched.value.child.admit.pipe(
                  Effect.timeoutFail({
                    duration: CANDIDATE_PARENT_RELEASE_TIMEOUT,
                    onTimeout: () => new AcnEnsuranceFailed({
                      reason: `Could not release parent channel for admitted ACN ${current.pid}`,
                    }),
                  }),
                )
                launched = Option.none()
              }
              const exactIdentity = yield* inspectProcess(processes, current.pid)
              const rootLive = Option.contains(exactIdentity, current.processStartIdentity)
              const treeAbsent = rootLive
                ? false
                : yield* processes.treeAbsent(exactFrom(current)).pipe(
                    Effect.mapError((error) => new AcnEnsuranceFailed({ reason: error.message })),
                  )
              if (!rootLive && !treeAbsent) {
                return { _tag: "SurvivingPredecessorTree", owner: current }
              }
              if (treeAbsent) {
                observedOwner = ""
                observedHealthState = ""
                return yield* classifyWithoutLiveOwner(now)
              }
              return yield* probeHealth(current).pipe(
                Effect.map((observed): ConvergenceState => ({
                  _tag: "ObservableOwner",
                  owner: current,
                  observed,
                  now,
                })),
              )
            }),
        })

        const completed = yield* Match.value(state).pipe(
          Match.tag("SurvivingPredecessorTree", ({ owner }) =>
            retireOwner(owner).pipe(Effect.as(Option.none<ReadyInstance>()))),
          Match.tag("ObservableOwner", ({ owner, observed, now }) =>
            Effect.gen(function* () {
              const currentOwnerKey = ownerKey(owner)
              const nextHealthState = Option.match(observed, {
                onNone: () => "Unavailable",
                onSome: ({ health }) => health.state._tag,
              })
              if (observedOwner !== currentOwnerKey || observedHealthState !== nextHealthState) {
                if (observedOwner !== currentOwnerKey) ownerObservedAt = now
                observedOwner = currentOwnerKey
                observedHealthState = nextHealthState
                observedHealthStateSince = now
              }
              if (Option.isSome(observed)) {
                const { health, status } = observed.value
                if (health.pid !== owner.pid ||
                  (status === 200) !== (health.state._tag === "Ready") ||
                  (status !== 200 && status !== 503)) {
                  yield* retireOwner(owner)
                  return Option.none<ReadyInstance>()
                }
                if (health.revision < target.revision) {
                  yield* prepare
                  yield* retireOwner(owner)
                  return Option.none<ReadyInstance>()
                }
                const progress = acnLifecycleObservationFromHealthState(health.state)
                if (Option.isSome(progress)) emit({ _tag: "Observation", observation: progress.value })
                const ready = yield* readyInstance(owner, observed.value)
                if (Option.isSome(ready)) return ready
                if (health.state._tag === "Starting" &&
                  now - ownerObservedAt >= Duration.toMillis(STARTUP_CEILING)) {
                  yield* retireOwner(owner)
                  return yield* new AcnEnsuranceFailed({
                    reason: "Magnitude daemon did not become ready within the startup deadline",
                  })
                }
              }
              // Live Starting health is bounded only by STARTUP_CEILING. Progress is
              // presentation-only and cannot establish liveness. Unobservable health
              // uses HEALTH_GRACE, while Stopping uses STOPPING_GRACE.
              const grace = Option.match(observed, {
                onNone: () => Option.some(HEALTH_GRACE),
                onSome: ({ health }) =>
                  health.state._tag === "Stopping"
                    ? Option.some(STOPPING_GRACE)
                    : health.state._tag === "Starting"
                      ? Option.none<Duration.Duration>()
                      : Option.some(HEALTH_GRACE),
              })
              if (Option.isSome(grace) &&
                now - observedHealthStateSince >= Duration.toMillis(grace.value)) {
                yield* retireOwner(owner)
              } else {
                yield* Effect.sleep(COORDINATION_POLL_INTERVAL)
              }
              return Option.none<ReadyInstance>()
            })),
          Match.tag("CandidatePending", () =>
            Effect.sleep(COORDINATION_POLL_INTERVAL).pipe(Effect.as(Option.none<ReadyInstance>()))),
          Match.tag("CandidateExited", ({ candidate, code, stderr }) => Effect.fail(new AcnEnsuranceFailed({
            reason: `ACN candidate ${candidate.process.pid} exited with code ${code} before admission${stderr ? `:\n${stderr}` : ""}`,
          }))),
          Match.tag("CandidateAdmissionExpired", ({ candidate }) => Effect.fail(new AcnEnsuranceFailed({
            reason: `ACN candidate ${candidate.process.pid} did not commit admission`,
          }))),
          Match.tag("LaunchOccurrenceLost", () => Effect.fail(new AcnEnsuranceFailed({
            reason: "Magnitude daemon exited during startup before it became ready",
          }))),
          Match.tag("LaunchCandidate", () => Effect.gen(function* () {
            const command = yield* prepare
            const argv = [
              ...command.command,
              ...(options.debug === true && !command.command.includes("--debug") ? ["--debug"] : []),
              "--parent-bound",
              "--data-dir",
              dataDirectory,
            ]
            if (!Arr.isNonEmptyReadonlyArray(argv)) {
              return yield* new AcnEnsuranceFailed({ reason: "Cannot spawn an empty ACN command" })
            }
            const child = yield* spawner.spawn(argv)
            hasLaunched = true
            const identity = yield* inspectProcess(processes, child.pid)
            if (Option.isNone(identity)) {
              const exit = yield* child.exited.pipe(
                Effect.timeoutOption(Duration.millis(100)),
              )
              return yield* new AcnEnsuranceFailed({
                reason: Option.match(exit, {
                  onNone: () => `Spawned ACN ${child.pid} exited before identity inspection`,
                  onSome: ({ code, stderr }) =>
                    `Spawned ACN ${child.pid} exited with code ${code} before identity inspection${stderr ? `:\n${stderr}` : ""}`,
                }),
              })
            }
            launched = Option.some({
              process: { pid: child.pid, processStartIdentity: identity.value },
              child,
              launchedAt: now,
            })
            return Option.none<ReadyInstance>()
          })),
          Match.exhaustive,
        )
        if (Option.isSome(completed)) return completed.value
      }
    })
    return Effect.scoped(run).pipe(Effect.timeoutFail({
      duration: ACN_ENSURE_TIMEOUT,
      onTimeout: () => new AcnEnsuranceFailed({
        reason: "ACN ensurance did not converge within its absolute deadline",
      }),
    }))
  }

  const ensure: AcnInstanceManager["ensure"] = (request) =>
    Stream.asyncPush<AcnEnsureEvent, AcnEnsuranceErrorType>((sink) =>
      Effect.forkScoped(ensureEffect(request.target, (event) => sink.single(event)).pipe(
        Effect.match({
          onFailure: sink.fail,
          onSuccess: (instance) => {
            sink.single({ _tag: "Ready", instance })
            sink.end()
          },
        }),
      )), { bufferSize: "unbounded" })

  const stop = Effect.gen(function* () {
    const owner = yield* retryStore(owners.current)
    if (Option.isNone(owner)) return
    yield* retireOwner(owner.value).pipe(
      Effect.mapError((error) => new AcnAdministrationFailed({ reason: error.reason })),
    )
  }).pipe(Effect.mapError((error) => error instanceof AcnAdministrationFailed
    ? error
    : new AcnAdministrationFailed({
        reason: error instanceof Error ? error.message : String(error),
      })))

  return AcnInstanceManager.of({ ensure, stop })
})

export const makeLocalAcnInstanceManager = (
  options: LocalAcnInstanceManagerOptions = {},
) => makeLocalAcnInstanceManagerWithProcessController(options).pipe(
  Effect.provideService(ExactProcessController, ExactProcessControllerLive),
)
