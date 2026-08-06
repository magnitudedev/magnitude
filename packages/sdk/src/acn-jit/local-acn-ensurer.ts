import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import {
  AcnReady,
  type AcnIdentity,
  type AcnInstanceId,
  AcnHealthResponseSchema,
  type AcnHealthResponse,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol"
import { canUseAcnIdentity, compareAcnIdentities } from "@magnitudedev/acn-protocol/acn-identity"
import {
  AcnProcessStateConflict,
  applyAcnProcessCommand,
  currentProcessStartIdentity,
  readAcnProcessState,
  readAcnProcessStateRevision,
  readProcessStartIdentity,
  type AcnProcessCommand,
  type AcnProcessRevision,
  type AcnProcessState,
  type AcnChangeResult,
  type AssignedAcn,
  type ExactAcnCandidate,
  type ExactIcnProcess,
  type ExactProcess,
} from "@magnitudedev/acn-protocol/process-state"
import type { ArtifactInstallationEvent } from "@magnitudedev/release"
import { Array as Arr, Clock, Context, Duration, Effect, Option, Schema, Scope, Stream } from "effect"
import { defaultDataDir, resolveBinaryCommand } from "../binary"
import type { BinaryAcquisitionEvent } from "../binary"
import {
  AcnEnsurer,
  type AcnEnsureEvent,
  type AcnEnsureRequest,
  type ReadyAcn,
} from "./acn-ensurer"
import {
  AcnDaemonAdministrator,
  type AcnDaemonAdministrator as AcnDaemonAdministratorService,
} from "./acn-daemon-administrator"
import { ChildProcessSpawner } from "./child-process"
import {
  AcnAdministrationFailed,
  AcnEnsuranceError,
  AcnEnsuranceFailed,
  type AcnEnsuranceError as AcnEnsuranceErrorType,
} from "./errors"
import {
  acnLifecycleObservationFromHealthState,
  acnStartupProgressKey,
} from "./lifecycle"

export interface AcnLaunchOverride {
  readonly identity: AcnIdentity
  readonly command: Arr.NonEmptyReadonlyArray<string>
}

export interface LocalAcnEnsurerOptions {
  readonly binaryPath?: string
  readonly dataDir?: string
  readonly debug?: boolean
  readonly launchOverride?: AcnLaunchOverride
}

export interface LocalAcnDaemonAdministratorOptions {
  readonly dataDir?: string
}

export type LocalAcnEnsurer = AcnEnsurer

const normalizeEnsuranceError = (error: unknown): AcnEnsuranceErrorType =>
  Schema.is(AcnEnsuranceError)(error)
    ? error
    : new AcnEnsuranceFailed({ reason: String(error) })

const sameProcess = (left: ExactProcess, right: ExactProcess): boolean =>
  left.pid === right.pid && left.processStartIdentity === right.processStartIdentity

const sameAcnOccurrence = (
  left: Pick<AssignedAcn, "id" | "pid" | "processStartIdentity">,
  right: Pick<AssignedAcn, "id" | "pid" | "processStartIdentity">,
): boolean => left.id === right.id && sameProcess(left, right)

const ASSIGNMENT_STALL = Duration.seconds(30)
const CHANGE_STALL = Duration.seconds(30)
const HEALTH_PROBE_TIMEOUT = Duration.seconds(2)
const PROCESS_EXIT_POLL_INTERVAL = Duration.millis(50)
const RECONCILIATION_POLL_INTERVAL = Duration.millis(50)
const STATE_OBSERVATION_INTERVAL = Duration.millis(100)

const stateRevision = (state: Option.Option<AcnProcessState>): Option.Option<AcnProcessRevision> =>
  Option.map(state, (value) => value.revision)

const processIsExact = (process: ExactProcess) =>
  readProcessStartIdentity(process.pid).pipe(
    Effect.map(Option.exists((identity) => identity === process.processStartIdentity)),
  )

const waitForExactExit = (process: ExactProcess, duration: Duration.DurationInput) =>
  Effect.gen(function* () {
    const deadline = (yield* Clock.currentTimeMillis) + Duration.toMillis(Duration.decode(duration))
    while ((yield* Clock.currentTimeMillis) < deadline) {
      if (!(yield* processIsExact(process))) return true
      yield* Effect.sleep(PROCESS_EXIT_POLL_INTERVAL)
    }
    return !(yield* processIsExact(process))
  })

const signalExact = (process: ExactProcess, signal: NodeJS.Signals) =>
  Effect.gen(function* () {
    if (!(yield* processIsExact(process))) return
    yield* Effect.try({
      try: () => globalThis.process.kill(process.pid, signal),
      catch: (error) => new AcnEnsuranceFailed({
        reason: `Failed to send ${signal} to exact process ${process.pid}: ${String(error)}`,
      }),
    }).pipe(Effect.catchIf(
      (error) => error.reason.includes("ESRCH"),
      () => Effect.void,
    ))
  })

const probeHealth = (
  current: AssignedAcn,
  timeout: Duration.DurationInput,
  client: HttpClient.HttpClient,
) => client.execute(HttpClientRequest.get(`${current.url}/health`)).pipe(
  Effect.timeout(timeout),
  Effect.flatMap((response) => response.json),
  Effect.flatMap(Schema.decodeUnknown(AcnHealthResponseSchema)),
  Effect.mapError((error) => new AcnEnsuranceFailed({
    reason: `Failed to observe ACN ${current.id}: ${String(error)}`,
  })),
  Effect.filterOrFail(
    (health) =>
      health.id === current.id &&
      health.version === current.identity &&
      health.pid === current.pid,
    () => new AcnEnsuranceFailed({
      reason: `ACN ${current.id} health does not match assigned process state`,
    }),
  ),
)

const readyFrom = (current: AssignedAcn): ReadyAcn => ({
  id: current.id,
  identity: current.identity,
  url: current.url,
  pid: current.pid,
  processStartIdentity: current.processStartIdentity,
  lifecycle: new AcnReady({}),
})

const artifactProgress = (
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>,
): AcnStartupProgress => ({
  completed: event.progress.acceptedBytes,
  totalBytes: event.progress.totalBytes,
  unit: "Bytes",
  attempt: Option.some(event.progress.attempt),
})

const emitHealth = (health: AcnHealthResponse, emit: (event: AcnEnsureEvent) => void) =>
  Option.match(acnLifecycleObservationFromHealthState(health.state), {
    onNone: () => Effect.void,
    onSome: (observation) => Effect.sync(() => emit({ _tag: "Observation", observation })),
  })

export interface PreparedCommand {
  readonly identity: AcnIdentity
  readonly command: Arr.NonEmptyReadonlyArray<string>
}

/** Host-local source of launch material for exact ACN identities. */
export interface AcnLaunchSource {
  readonly supports: (identity: AcnIdentity) => boolean
  readonly prepare: (
    identity: AcnIdentity,
    emit: (event: AcnEnsureEvent) => void,
  ) => Effect.Effect<PreparedCommand, AcnEnsuranceErrorType>
}

export const AcnLaunchSource = Context.GenericTag<AcnLaunchSource>(
  "@magnitudedev/sdk/AcnLaunchSource",
)

type ChangeObservation =
  | { readonly _tag: "Completed" }
  | { readonly _tag: "Failed"; readonly reason: string }
  | { readonly _tag: "Stalled"; readonly state: AcnProcessState }

type ReadinessObservation =
  | { readonly _tag: "Ready"; readonly instance: ReadyAcn }
  | { readonly _tag: "Replace"; readonly current: AssignedAcn }
  | { readonly _tag: "StartupFailed"; readonly reason: string }
  | { readonly _tag: "Follow"; readonly changeRevision: AcnProcessRevision }
  | { readonly _tag: "Reclassify" }

const makeLocalTerminationKernel = (
  options: LocalAcnDaemonAdministratorOptions,
) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const client = yield* HttpClient.HttpClient
  const commandExecutor = yield* CommandExecutor.CommandExecutor
  const dataDirectory = options.dataDir ?? defaultDataDir()
  const manager: ExactProcess = {
    pid: process.pid,
    processStartIdentity: yield* currentProcessStartIdentity.pipe(Effect.orDie),
  }

  const provideLocal = <A, E, R>(effect: Effect.Effect<A, E, R>) => effect.pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
  )

  const readState = provideLocal(readAcnProcessState(dataDirectory))

  const apply = (state: Option.Option<AcnProcessState>, command: AcnProcessCommand) =>
    provideLocal(applyAcnProcessCommand({
      dataDirectory,
      expectedRevision: stateRevision(state),
      command,
    }))

  const stillOwnsRevision = (revision: AcnProcessRevision) => readState.pipe(
    Effect.map(Option.exists((state) =>
      state.revision === revision &&
      state.mode._tag === "Changing" &&
      state.mode.owner._tag === "Manager" &&
      sameProcess(state.mode.owner.process, manager),
    )),
  )

  const signalWhileOwned = (
    revision: AcnProcessRevision,
    exactProcess: ExactProcess,
    signal: NodeJS.Signals,
  ) => Effect.gen(function* () {
    if (!(yield* stillOwnsRevision(revision))) return false
    yield* provideLocal(signalExact(exactProcess, signal))
    return true
  })

  const stopExact = (
    state: AcnProcessState,
    exactProcess: ExactProcess,
    requestShutdown: Effect.Effect<unknown, unknown> = Effect.void,
  ) => Effect.gen(function* () {
    if (!(yield* stillOwnsRevision(state.revision))) return false
    yield* requestShutdown.pipe(Effect.timeout(Duration.millis(750)), Effect.ignore)
    if (yield* provideLocal(waitForExactExit(exactProcess, Duration.seconds(2)))) return true
    if (!(yield* signalWhileOwned(state.revision, exactProcess, "SIGINT"))) return false
    if (yield* provideLocal(waitForExactExit(exactProcess, Duration.seconds(2)))) return true
    if (!(yield* signalWhileOwned(state.revision, exactProcess, "SIGKILL"))) return false
    return yield* provideLocal(waitForExactExit(exactProcess, Duration.seconds(2)))
  })

  const stopIcn = (state: AcnProcessState, icn: ExactIcnProcess) =>
    Effect.gen(function* () {
      if (!(yield* provideLocal(processIsExact(icn)))) return true
      if (!(yield* signalWhileOwned(state.revision, icn, "SIGTERM"))) return false
      if (yield* provideLocal(waitForExactExit(icn, Duration.seconds(1)))) return true
      if (!(yield* signalWhileOwned(state.revision, icn, "SIGKILL"))) return false
      return yield* provideLocal(waitForExactExit(icn, Duration.seconds(1)))
    })

  const retireAssigned = (state: AcnProcessState, current: AssignedAcn) =>
    Effect.gen(function* () {
      const acnExited = yield* stopExact(
        state,
        current,
        client.execute(
          HttpClientRequest.post(`${current.url}/shutdown`).pipe(
            HttpClientRequest.setHeader("x-magnitude-acn-id", current.id),
          ),
        ),
      )
      if (!acnExited) return false
      if (Option.isSome(current.ownedIcn) && !(yield* stopIcn(state, current.ownedIcn.value))) return false
      return true
    })

  const stopCurrent = Effect.gen(function* () {
    let observedRevision = Option.none<AcnProcessRevision>()
    let unchangedSince = yield* Clock.currentTimeMillis

    while (true) {
      const stateOption = yield* readState
      if (Option.isNone(stateOption) || stateOption.value.mode._tag === "Unassigned") return
      const state = stateOption.value

      if (state.mode._tag === "Assigned") {
        const changed = yield* apply(stateOption, {
          _tag: "BeginTerminateCurrent",
          manager,
        }).pipe(Effect.either)
        if (changed._tag === "Left") {
          if (changed.left instanceof AcnProcessStateConflict) continue
          return yield* changed.left
        }
        observedRevision = Option.none()
        unchangedSince = yield* Clock.currentTimeMillis
        continue
      }

      if (state.mode._tag === "Unassigned") return
      const changing = state.mode
      if (changing.purpose._tag === "Ensure") {
        const changed = yield* apply(stateOption, {
          _tag: "BeginTerminateCurrent",
          manager,
        }).pipe(Effect.either)
        if (changed._tag === "Left") {
          if (changed.left instanceof AcnProcessStateConflict) continue
          return yield* changed.left
        }
        observedRevision = Option.none()
        unchangedSince = yield* Clock.currentTimeMillis
        continue
      }

      if (changing.owner._tag === "Manager" && sameProcess(changing.owner.process, manager)) {
        const phase = changing.owner.phase
        if (phase._tag === "BlockedCandidateCleanup") {
          yield* apply(stateOption, { _tag: "RetryCandidateCleanup", manager }).pipe(
            Effect.catchTag("AcnProcessStateConflict", () => Effect.void),
          )
          continue
        }
        if (phase._tag === "Spawning" || phase._tag === "Preparing") {
          return yield* new AcnEnsuranceFailed({
            reason: `Terminate change cannot retain ${phase._tag}`,
          })
        }

        const retired = phase._tag === "RetiringAssigned"
          ? yield* retireAssigned(state, phase.current)
          : yield* stopExact(state, phase.candidate)
        if (!retired) {
          return yield* new AcnEnsuranceFailed({
            reason: phase._tag === "RetiringAssigned"
              ? `Could not prove ACN ${phase.current.id} and its ICN absent`
              : `Could not prove candidate ACN ${phase.candidate.pid} absent`,
          })
        }
        yield* apply(
          stateOption,
          phase._tag === "RetiringAssigned"
            ? { _tag: "PredecessorExited", manager }
            : { _tag: "CandidateExited", manager },
        ).pipe(Effect.catchTag("AcnProcessStateConflict", () => Effect.void))
        continue
      }

      if (!Option.contains(observedRevision, state.revision)) {
        observedRevision = Option.some(state.revision)
        unchangedSince = yield* Clock.currentTimeMillis
      }
      const owner = changing.owner._tag === "Manager"
        ? changing.owner.process
        : changing.owner.candidate
      const ownerAlive = yield* provideLocal(processIsExact(owner))
      if (!ownerAlive || (yield* Clock.currentTimeMillis) - unchangedSince >= Duration.toMillis(CHANGE_STALL)) {
        yield* apply(stateOption, { _tag: "TakeOver", manager }).pipe(
          Effect.catchTag("AcnProcessStateConflict", () => Effect.void),
        )
        continue
      }
      yield* Effect.sleep(STATE_OBSERVATION_INTERVAL)
    }
  }).pipe(Effect.mapError((error) => new AcnAdministrationFailed({
    reason: typeof error === "object" && error !== null && "reason" in error
      ? String(error.reason)
      : String(error),
  })))

  return {
    fs,
    client,
    commandExecutor,
    dataDirectory,
    manager,
    provideLocal,
    readState,
    apply,
    stopExact,
    retireAssigned,
    stopCurrent,
  }
})

export const makeLocalAcnDaemonAdministrator = (
  options: LocalAcnDaemonAdministratorOptions = {},
): Effect.Effect<
  AcnDaemonAdministratorService,
  never,
  FileSystem.FileSystem | HttpClient.HttpClient | CommandExecutor.CommandExecutor
> => makeLocalTerminationKernel(options).pipe(
  Effect.map((kernel) => AcnDaemonAdministrator.of({ stopCurrent: kernel.stopCurrent })),
)

/** Resolves one exact proxy target without health probing or lifecycle projection. */
export const resolveAssignedAcnProxyTarget = (
  dataDirectory: string,
  expectedId: AcnInstanceId,
): Effect.Effect<Option.Option<string>, AcnEnsuranceFailed, FileSystem.FileSystem> =>
  readAcnProcessState(dataDirectory).pipe(
    Effect.map((state) => Option.flatMap(state, (value) =>
      value.mode._tag === "Assigned" && value.mode.current.id === expectedId
        ? Option.some(value.mode.current.url)
        : Option.none(),
    )),
    Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
  )

/** Local implementation of the exact-current ACN process contract. */
export const makeLocalAcnEnsurer = (
  options: LocalAcnEnsurerOptions = {},
): Effect.Effect<
  LocalAcnEnsurer,
  never,
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor
  | Path.Path
  | ChildProcessSpawner
  | Scope.Scope
> => Effect.gen(function* () {
  const termination = yield* makeLocalTerminationKernel(options)
  const {
    fs,
    client,
    commandExecutor,
    dataDirectory,
    manager,
    provideLocal,
    readState,
    apply,
    stopExact,
    retireAssigned,
  } = termination
  const path = yield* Path.Path
  const spawner = yield* ChildProcessSpawner
  const operationScope = yield* Effect.scope
  const reconcilePermit = yield* Effect.makeSemaphore(1)

  const resolveCommand = (
    identity: AcnIdentity,
    emit: (event: AcnEnsureEvent) => void,
  ): Effect.Effect<PreparedCommand, AcnEnsuranceErrorType> =>
    options.launchOverride !== undefined && options.launchOverride.identity === identity
      ? Effect.succeed({ identity, command: options.launchOverride.command })
      : (() => {
        let plan: Option.Option<{
          readonly daemonBytes: number
          readonly inferenceEngineBytes: number
          readonly inferenceEngineBytesExact: boolean
        }> = Option.none()
        const report = (event: BinaryAcquisitionEvent) => Effect.sync(() => {
          if (event._tag === "Planned") {
            plan = Option.some(event.plan)
          } else if (event.event._tag === "Downloading" && Option.isSome(plan)) {
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
          version: identity,
          dataDir: dataDirectory,
          acquisitionObserver: Option.some({ report }),
        }).pipe(
          Effect.provideService(FileSystem.FileSystem, fs),
          Effect.provideService(HttpClient.HttpClient, client),
          Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
          Effect.provideService(Path.Path, path),
          Effect.map((resolved) => ({ identity, command: resolved.command })),
        )
      })().pipe(Effect.mapError(normalizeEnsuranceError))

  const defaultLaunchSource: AcnLaunchSource = {
    supports: (identity) =>
      options.launchOverride?.identity === identity || !identity.includes("+dev."),
    prepare: resolveCommand,
  }
  const launchSource = Option.getOrElse(
    yield* Effect.serviceOption(AcnLaunchSource),
    () => defaultLaunchSource,
  )

  const spawnCandidate = (
    state: AcnProcessState,
    prepared: PreparedCommand,
  ) => Effect.scoped(Effect.gen(function* () {
    if (
      state.mode._tag !== "Changing" ||
      state.mode.purpose._tag !== "Ensure" ||
      state.mode.owner._tag !== "Manager" ||
      state.mode.owner.phase._tag !== "Spawning" ||
      state.mode.purpose.target !== prepared.identity
    ) return
    const argv = [
      ...prepared.command,
      ...(options.debug === true && !prepared.command.includes("--debug") ? ["--debug"] : []),
      "--change-revision",
      String(state.mode.changeRevision),
      "--data-dir",
      dataDirectory,
    ]
    if (!Arr.isNonEmptyReadonlyArray(argv)) {
      yield* apply(Option.some(state), {
        _tag: "FailSpawning",
        manager,
        reason: "Cannot spawn an empty ACN command",
      })
      return
    }
    const child = yield* spawner.spawn(argv)
    const pid = child.pid
    const startIdentity = yield* provideLocal(readProcessStartIdentity(pid)).pipe(
      Effect.flatMap(Option.match({
        onNone: () => Effect.fail(new AcnEnsuranceFailed({ reason: `Spawned ACN ${pid} exited before publication` })),
        onSome: Effect.succeed,
      })),
    )
    const candidate: ExactAcnCandidate = {
      identity: state.mode.purpose.target,
      pid,
      processStartIdentity: startIdentity,
    }
    const transferred = yield* apply(Option.some(state), {
      _tag: "CandidateSpawned",
      manager,
      candidate,
    }).pipe(Effect.either)
    if (transferred._tag === "Left") {
      if (!(transferred.left instanceof AcnProcessStateConflict)) return yield* transferred.left
      return
    }
    yield* child.handoff
  })).pipe(
    Effect.catchAll((error) => apply(Option.some(state), {
      _tag: "FailSpawning",
      manager,
      reason: String(error),
    }).pipe(Effect.ignore)),
  )

  const reconcileOwned = (
    state: AcnProcessState,
    prepared: Option.Option<PreparedCommand>,
    emit: (event: AcnEnsureEvent) => void,
  ) => Effect.gen(function* () {
      const fresh = yield* readState
      if (Option.isNone(fresh) || fresh.value.revision !== state.revision) return prepared
      const current = fresh.value
      if (
        current.mode._tag !== "Changing" ||
        current.mode.owner._tag !== "Manager" ||
        !sameProcess(current.mode.owner.process, manager)
      ) return prepared
      switch (current.mode.owner.phase._tag) {
        case "Preparing": {
          if (current.mode.purpose._tag !== "Ensure") return Option.none()
          const target = current.mode.purpose.target
          if (!launchSource.supports(target)) {
            yield* apply(Option.some(current), {
              _tag: "PreparationFailed",
              manager,
              reason: `This host has no launch source for ACN ${target}`,
            }).pipe(Effect.catchTag("AcnProcessStateConflict", () => Effect.void))
            return Option.none()
          }
          const resolved = yield* launchSource.prepare(target, emit).pipe(Effect.either)
          if (resolved._tag === "Left") {
            const reason = "reason" in resolved.left ? String(resolved.left.reason) : String(resolved.left)
            yield* apply(Option.some(current), {
              _tag: "PreparationFailed",
              manager,
              reason,
            }).pipe(Effect.catchTag("AcnProcessStateConflict", () => Effect.void))
            return Option.none()
          }
          const advanced = yield* apply(Option.some(current), {
            _tag: "PreparationSucceeded",
            manager,
          }).pipe(Effect.either)
          return advanced._tag === "Right" ? Option.some(resolved.right) : Option.none()
        }
        case "RetiringAssigned": {
          const retired = yield* retireAssigned(current, current.mode.owner.phase.current)
          yield* apply(Option.some(current), retired
            ? { _tag: "PredecessorExited", manager }
            : {
                _tag: "RetirementBlocked",
                manager,
                reason: `Could not prove ACN ${current.mode.owner.phase.current.id} and its ICN absent`,
              }).pipe(Effect.catchTag("AcnProcessStateConflict", () => Effect.void))
          return prepared
        }
        case "RetiringCandidate": {
          const candidate = current.mode.owner.phase.candidate
          const retired = yield* stopExact(current, candidate)
          yield* apply(Option.some(current), retired
            ? { _tag: "CandidateExited", manager }
            : {
                _tag: "CandidateCleanupBlocked",
                manager,
                reason: `Could not prove candidate ACN ${candidate.pid} absent`,
              }).pipe(
            Effect.catchTag("AcnProcessStateConflict", () => Effect.void),
          )
          return Option.none()
        }
        case "BlockedCandidateCleanup":
          return prepared
        case "Spawning": {
          if (Option.isNone(prepared)) {
            yield* apply(Option.some(current), {
              _tag: "FailSpawning",
              manager,
              reason: "Launch preparation was lost before spawning",
            }).pipe(Effect.catchTag("AcnProcessStateConflict", () => Effect.void))
            return Option.none()
          }
          yield* spawnCandidate(current, prepared.value)
          return prepared
        }
      }
    })

  const reconcileChange = (
    changeRevision: AcnProcessRevision,
    emit: (event: AcnEnsureEvent) => void,
  ) => reconcilePermit.withPermits(1)(Effect.gen(function* () {
    let prepared = Option.none<PreparedCommand>()
    while (true) {
      const stateOption = yield* readState
      if (Option.isNone(stateOption)) return
      const state = stateOption.value
      if (state.mode._tag !== "Changing" || state.mode.changeRevision !== changeRevision) return
      if (state.mode.owner._tag === "Manager" && sameProcess(state.mode.owner.process, manager)) {
        if (state.mode.owner.phase._tag === "BlockedCandidateCleanup") return
        prepared = yield* reconcileOwned(state, prepared, emit)
        yield* Effect.sleep(RECONCILIATION_POLL_INTERVAL)
        continue
      }
      // CandidateSpawned transfers reconciliation to the candidate itself.
      // Any other owner transfer fences this supervisor; foreground ensure
      // observers own dead/stalled-owner recovery.
      return
    }
  }))

  const applyAndReconcile = (
    state: Option.Option<AcnProcessState>,
    command: AcnProcessCommand,
    emit: (event: AcnEnsureEvent) => void,
  ) => Effect.uninterruptible(
    Effect.gen(function* () {
      const changed = yield* apply(state, command)
      if (changed.mode._tag === "Changing") {
        yield* Effect.forkIn(reconcileChange(changed.mode.changeRevision, emit), operationScope)
      }
      return changed
    }),
  )

  const resultForChange = (
    changeRevision: AcnProcessRevision,
    latestRevision: AcnProcessRevision,
  ): Effect.Effect<Option.Option<AcnChangeResult>, AcnEnsuranceErrorType> =>
    Effect.gen(function* () {
      for (let revision = changeRevision + 1; revision <= latestRevision; revision += 1) {
        const historical = yield* provideLocal(readAcnProcessStateRevision(
          dataDirectory,
          revision as AcnProcessRevision,
        ))
        if (historical.mode._tag === "Changing") continue
        const result = historical.mode.result
        if (Option.exists(result, (value) => value.changeRevision === changeRevision)) return result
      }
      return Option.none()
    }).pipe(Effect.mapError(normalizeEnsuranceError))

  const assignmentForChange = (
    changeRevision: AcnProcessRevision,
    latestRevision: AcnProcessRevision,
  ): Effect.Effect<Option.Option<AssignedAcn>, AcnEnsuranceErrorType> =>
    Effect.gen(function* () {
      for (let revision = changeRevision + 1; revision <= latestRevision; revision += 1) {
        const historical = yield* provideLocal(readAcnProcessStateRevision(
          dataDirectory,
          revision as AcnProcessRevision,
        ))
        if (historical.mode._tag !== "Assigned") continue
        const result = Option.getOrUndefined(historical.mode.result)
        if (result?._tag === "Admitted" && result.changeRevision === changeRevision) {
          return Option.some(historical.mode.current)
        }
      }
      return Option.none()
    }).pipe(Effect.mapError(normalizeEnsuranceError))

  const observeChange = (initial: AcnProcessState): Effect.Effect<ChangeObservation, AcnEnsuranceErrorType> =>
    Effect.gen(function* () {
      if (initial.mode._tag !== "Changing") return { _tag: "Completed" } as const
      const changeRevision = initial.mode.changeRevision
      let observedRevision = initial.revision
      let unchangedSince = yield* Clock.currentTimeMillis
      while (true) {
        const observed = yield* readState
        if (Option.isNone(observed)) {
          return yield* new AcnEnsuranceFailed({ reason: `ACN change ${changeRevision} disappeared` })
        }
        const state = observed.value
        if (state.mode._tag !== "Changing" || state.mode.changeRevision !== changeRevision) {
          const result = Option.getOrUndefined(yield* resultForChange(changeRevision, state.revision))
          if (result === undefined) {
            return yield* new AcnEnsuranceFailed({
              reason: `ACN change ${changeRevision} was superseded without a durable result`,
            })
          }
          return result._tag === "Failed"
            ? { _tag: "Failed", reason: result.reason } as const
            : { _tag: "Completed" } as const
        }
        if (
          state.mode.owner._tag === "Manager" &&
          state.mode.owner.phase._tag === "BlockedCandidateCleanup"
        ) {
          return { _tag: "Failed", reason: state.mode.owner.phase.reason } as const
        }
        if (state.revision !== observedRevision) {
          observedRevision = state.revision
          unchangedSince = yield* Clock.currentTimeMillis
        }
        const owner = state.mode.owner._tag === "Manager"
          ? state.mode.owner.process
          : state.mode.owner.candidate
        const alive = yield* provideLocal(processIsExact(owner))
        if (!alive && state.mode.owner._tag === "Candidate") {
          const failed = yield* apply(Option.some(state), {
            _tag: "CandidateFailed",
            candidate: state.mode.owner.candidate,
            reason: `Candidate ACN ${state.mode.owner.candidate.pid} exited before admission`,
          }).pipe(Effect.either)
          if (failed._tag === "Right") {
            return { _tag: "Failed", reason: failed.right.mode._tag === "Unassigned"
              ? Option.match(failed.right.mode.result, {
                  onNone: () => "Candidate exited before admission",
                  onSome: (result) => result._tag === "Failed" ? result.reason : "Candidate exited before admission",
                })
              : "Candidate exited before admission" } as const
          }
          if (!(failed.left instanceof AcnProcessStateConflict)) return yield* failed.left
          continue
        }
        if (!alive || (yield* Clock.currentTimeMillis) - unchangedSince >= Duration.toMillis(CHANGE_STALL)) {
          return { _tag: "Stalled", state } as const
        }
        yield* Effect.sleep(STATE_OBSERVATION_INTERVAL)
      }
    }).pipe(Effect.mapError(normalizeEnsuranceError))

  const waitUntilReady = (
    targetIdentity: AcnIdentity,
    emit: (event: AcnEnsureEvent) => void,
    startupChange: Option.Option<AcnProcessRevision>,
  ): Effect.Effect<ReadinessObservation, AcnEnsuranceErrorType> => Effect.gen(function* () {
    let bound = Option.none<AssignedAcn>()
    let progressKey: string | undefined
    let progressDeadline = (yield* Clock.currentTimeMillis) + Duration.toMillis(ASSIGNMENT_STALL)
    while (true) {
      const state = yield* readState
      if (Option.isNone(state)) {
        return Option.isSome(bound) || Option.isSome(startupChange)
          ? { _tag: "StartupFailed", reason: "ACN assignment disappeared during startup" } as const
          : { _tag: "Reclassify" } as const
      }
      if (state.value.mode._tag === "Changing") {
        if (Option.isSome(bound) || Option.isSome(startupChange)) {
          return state.value.mode.purpose._tag === "Ensure" &&
            canUseAcnIdentity(targetIdentity, state.value.mode.purpose.target)
            ? { _tag: "Follow", changeRevision: state.value.mode.changeRevision } as const
            : { _tag: "StartupFailed", reason: "Bound ACN startup occurrence was replaced" } as const
        }
        return { _tag: "Reclassify" } as const
      }
      if (state.value.mode._tag === "Unassigned") {
        const result = Option.getOrUndefined(state.value.mode.result)
        return Option.isSome(bound) || Option.isSome(startupChange)
          ? { _tag: "StartupFailed", reason: result?._tag === "Failed"
            ? result.reason
            : "ACN startup ended without assignment" } as const
          : { _tag: "Reclassify" } as const
      }
      const current = state.value.mode.current
      if (Option.isSome(bound) && !sameAcnOccurrence(bound.value, current)) {
        const result = Option.getOrUndefined(state.value.mode.result)
        if (
          result?._tag === "Admitted" &&
          canUseAcnIdentity(targetIdentity, current.identity)
        ) return { _tag: "Follow", changeRevision: result.changeRevision } as const
        return { _tag: "StartupFailed", reason: "Bound ACN startup occurrence changed" } as const
      }
      if (Option.isSome(startupChange)) {
        const result = Option.getOrUndefined(state.value.mode.result)
        if (result?._tag !== "Admitted" || result.changeRevision !== startupChange.value) {
          const historical = yield* assignmentForChange(startupChange.value, state.value.revision)
          if (Option.isNone(historical) || !sameAcnOccurrence(historical.value, current)) {
            if (
              result?._tag === "Admitted" &&
              canUseAcnIdentity(targetIdentity, current.identity)
            ) return { _tag: "Follow", changeRevision: result.changeRevision } as const
            return { _tag: "StartupFailed", reason: `ACN change ${startupChange.value} no longer owns the assignment` } as const
          }
        }
        bound = Option.some(current)
      }
      if (!canUseAcnIdentity(targetIdentity, current.identity)) {
        return yield* new AcnEnsuranceFailed({ reason: `Assigned ACN ${current.identity} does not satisfy ${targetIdentity}` })
      }
      if (!(yield* provideLocal(processIsExact(current)))) {
        return Option.isSome(bound)
          ? { _tag: "StartupFailed", reason: `ACN ${current.id} exited before becoming ready` } as const
          : { _tag: "Replace", current } as const
      }
      const healthResult = yield* Effect.either(probeHealth(current, HEALTH_PROBE_TIMEOUT, client))
      if (healthResult._tag === "Left") {
        if ((yield* Clock.currentTimeMillis) >= progressDeadline) {
          return Option.isSome(bound)
            ? { _tag: "StartupFailed", reason: `ACN ${current.id} did not become ready` } as const
            : { _tag: "Replace", current } as const
        }
        yield* Effect.sleep(STATE_OBSERVATION_INTERVAL)
        continue
      }
      const health = healthResult.right
      yield* emitHealth(health, emit)
      if (health.state._tag === "Ready") {
        const confirmed = yield* readState
        if (
          Option.isSome(confirmed) &&
          confirmed.value.mode._tag === "Assigned" &&
          sameAcnOccurrence(confirmed.value.mode.current, current) &&
          confirmed.value.mode.current.identity === current.identity &&
          confirmed.value.mode.current.url === current.url
        ) return { _tag: "Ready", instance: readyFrom(current) } as const
        continue
      }
      if (health.state._tag === "Stopping") {
        return Option.isSome(bound)
          ? { _tag: "StartupFailed", reason: `ACN ${current.id} stopped before becoming ready` } as const
          : { _tag: "Replace", current } as const
      }
      bound = Option.some(current)
      const nextKey = acnStartupProgressKey(health.state)
      if (nextKey !== progressKey) {
        progressKey = nextKey
        progressDeadline = (yield* Clock.currentTimeMillis) + Duration.toMillis(ASSIGNMENT_STALL)
      } else if ((yield* Clock.currentTimeMillis) >= progressDeadline) {
        return { _tag: "StartupFailed", reason: `ACN ${current.id} startup stalled` } as const
      }
      yield* Effect.sleep(STATE_OBSERVATION_INTERVAL)
    }
  }).pipe(Effect.mapError(normalizeEnsuranceError))

  const targetFor = (
    requestIdentity: AcnIdentity,
    state: Option.Option<AcnProcessState>,
  ): AcnIdentity => Option.match(state, {
    onNone: () => requestIdentity,
    onSome: (current) => compareAcnIdentities(requestIdentity, current.identityFloor) >= 0
      ? requestIdentity
      : current.identityFloor,
  })

  const ensure = (request: AcnEnsureRequest, emit: (event: AcnEnsureEvent) => void) =>
    Effect.gen(function* () {
      let forced = Option.none<Pick<AssignedAcn, "id" | "pid" | "processStartIdentity">>()
      let stalledRevision = Option.none<AcnProcessRevision>()
      let joinedChange = Option.none<AcnProcessRevision>()

      while (true) {
        const state = yield* readState

        if (Option.isSome(joinedChange)) {
          if (Option.isNone(state)) {
            return yield* new AcnEnsuranceFailed({ reason: `ACN change ${joinedChange.value} disappeared` })
          }
          const currentState = state.value
          if (
            currentState.mode._tag === "Changing" &&
            currentState.mode.changeRevision === joinedChange.value
          ) {
            const observation = yield* observeChange(currentState)
            if (observation._tag === "Failed") {
              const restored = yield* readState
              if (
                Option.isSome(restored) &&
                restored.value.mode._tag === "Assigned" &&
                canUseAcnIdentity(request.minimumIdentity, restored.value.mode.current.identity)
              ) {
                joinedChange = Option.none()
                continue
              }
              return yield* new AcnEnsuranceFailed({ reason: observation.reason })
            }
            if (observation._tag === "Stalled") {
              const target = currentState.mode.purpose._tag === "Ensure"
                ? currentState.mode.purpose.target
                : undefined
              if (target !== undefined && launchSource.supports(target)) {
                const taken = yield* applyAndReconcile(
                  state,
                  { _tag: "TakeOver", manager },
                  emit,
                ).pipe(Effect.either)
                if (taken._tag === "Left" && !(taken.left instanceof AcnProcessStateConflict)) {
                  return yield* taken.left
                }
              } else {
                yield* Effect.sleep(STATE_OBSERVATION_INTERVAL)
              }
            }
            continue
          }
          const result = Option.getOrUndefined(yield* resultForChange(joinedChange.value, currentState.revision))
          if (result === undefined) {
            return yield* new AcnEnsuranceFailed({
              reason: `ACN change ${joinedChange.value} ended without a durable result`,
            })
          }
          if (result._tag === "Failed") {
            if (
              currentState.mode._tag === "Assigned" &&
              canUseAcnIdentity(request.minimumIdentity, currentState.mode.current.identity)
            ) {
              joinedChange = Option.none()
              continue
            }
            return yield* new AcnEnsuranceFailed({ reason: result.reason })
          }
          if (result._tag === "Admitted" && currentState.mode._tag === "Changing") {
            if (
              currentState.mode.purpose._tag === "Ensure" &&
              canUseAcnIdentity(request.minimumIdentity, currentState.mode.purpose.target)
            ) {
              joinedChange = Option.some(currentState.mode.changeRevision)
              continue
            }
          }
          if (result._tag === "Admitted" && currentState.mode._tag === "Assigned") {
            const currentResult = Option.getOrUndefined(currentState.mode.result)
            if (currentResult?._tag === "Admitted" && currentResult.changeRevision !== joinedChange.value) {
              if (canUseAcnIdentity(request.minimumIdentity, currentState.mode.current.identity)) {
                joinedChange = Option.some(currentResult.changeRevision)
                continue
              }
            }
          }
          if (result._tag !== "Admitted" || currentState.mode._tag !== "Assigned") {
            return yield* new AcnEnsuranceFailed({ reason: `ACN change ${joinedChange.value} did not admit an ACN` })
          }
          const readiness = yield* waitUntilReady(request.minimumIdentity, emit, joinedChange)
          if (readiness._tag === "Ready") return readiness.instance
          if (readiness._tag === "Follow") {
            joinedChange = Option.some(readiness.changeRevision)
            continue
          }
          return yield* new AcnEnsuranceFailed({
            reason: readiness._tag === "StartupFailed"
              ? readiness.reason
              : `ACN change ${joinedChange.value} lost its admitted occurrence`,
          })
        }

        if (Option.isSome(state) && state.value.mode._tag === "Assigned") {
          const current = state.value.mode.current
          const forceCurrent = Option.exists(forced, (replace) => sameAcnOccurrence(replace, current))
          if (!forceCurrent && canUseAcnIdentity(request.minimumIdentity, current.identity)) {
            const readiness = yield* waitUntilReady(request.minimumIdentity, emit, Option.none())
            if (readiness._tag === "Ready") return readiness.instance
            if (readiness._tag === "Reclassify") continue
            if (readiness._tag === "Follow") {
              joinedChange = Option.some(readiness.changeRevision)
              continue
            }
            if (readiness._tag === "StartupFailed") {
              return yield* new AcnEnsuranceFailed({ reason: readiness.reason })
            }
            forced = Option.some(readiness.current)
            stalledRevision = Option.none()
            continue
          }
        }

        if (Option.isSome(state) && state.value.mode._tag === "Changing") {
          const changing = state.value
          const mode = changing.mode
          if (mode._tag !== "Changing") continue
          if (mode.purpose._tag === "Terminate") {
            if (Option.contains(stalledRevision, changing.revision)) {
              const taken = yield* applyAndReconcile(
                state,
                { _tag: "TakeOver", manager },
                emit,
              ).pipe(Effect.either)
              if (taken._tag === "Left" && !(taken.left instanceof AcnProcessStateConflict)) {
                return yield* taken.left
              }
              stalledRevision = Option.none()
              continue
            }
            const observation = yield* observeChange(changing)
            if (observation._tag === "Failed") {
              return yield* new AcnEnsuranceFailed({ reason: observation.reason })
            }
            stalledRevision = observation._tag === "Stalled"
              ? Option.some(observation.state.revision)
              : Option.none()
            continue
          }

          if (
            mode.owner._tag === "Manager" &&
            mode.owner.phase._tag === "BlockedCandidateCleanup" &&
            launchSource.supports(mode.purpose.target)
          ) {
            const retrying = yield* applyAndReconcile(
              state,
              { _tag: "RetryCandidateCleanup", manager },
              emit,
            ).pipe(Effect.either)
            if (retrying._tag === "Left") {
              if (retrying.left instanceof AcnProcessStateConflict) continue
              return yield* retrying.left
            }
            joinedChange = Option.some(mode.changeRevision)
            continue
          }

          if (canUseAcnIdentity(request.minimumIdentity, mode.purpose.target)) {
            joinedChange = Option.some(mode.changeRevision)
            continue
          }

          const targetIdentity = targetFor(request.minimumIdentity, state)
          if (
            mode.owner._tag === "Manager" &&
            mode.owner.phase._tag === "Preparing" &&
            launchSource.supports(targetIdentity)
          ) {
            const upgraded = yield* applyAndReconcile(
              state,
              { _tag: "UpgradeEnsure", target: targetIdentity, manager },
              emit,
            ).pipe(Effect.either)
            if (upgraded._tag === "Left") {
              if (upgraded.left instanceof AcnProcessStateConflict) continue
              return yield* upgraded.left
            }
            joinedChange = Option.some(mode.changeRevision)
            continue
          }

          const observation = yield* observeChange(changing)
          if (observation._tag === "Failed") {
            stalledRevision = Option.none()
          } else {
            stalledRevision = observation._tag === "Stalled"
              ? Option.some(observation.state.revision)
              : Option.none()
          }
          continue
        }

        const targetIdentity = targetFor(request.minimumIdentity, state)
        if (!launchSource.supports(targetIdentity)) {
          return yield* new AcnEnsuranceFailed({
            reason: `This client cannot launch required ACN ${targetIdentity}`,
          })
        }

        let transition: AcnProcessCommand | undefined
        if (Option.isNone(state) || state.value.mode._tag === "Unassigned") {
          transition = { _tag: "BeginEnsure", target: targetIdentity, manager }
        } else if (state.value.mode._tag === "Assigned") {
          const current = state.value.mode.current
          const forceCurrent = Option.exists(forced, (replace) => sameAcnOccurrence(replace, current))
          if (!forceCurrent && canUseAcnIdentity(targetIdentity, current.identity)) continue
          transition = { _tag: "BeginReplacement", target: targetIdentity, manager, current }
        } else {
          continue
        }

        const changed = yield* applyAndReconcile(
          state,
          transition,
          emit,
        ).pipe(Effect.either)
        if (changed._tag === "Left") {
          if (changed.left instanceof AcnProcessStateConflict) continue
          return yield* changed.left
        }
        if (changed.right.mode._tag === "Changing" && changed.right.mode.purpose._tag === "Ensure") {
          joinedChange = Option.some(changed.right.mode.changeRevision)
        }
        stalledRevision = Option.none()
      }
    }).pipe(Effect.mapError(normalizeEnsuranceError))

  const ensureStream: AcnEnsurer["ensure"] = (request) =>
    Stream.asyncPush<AcnEnsureEvent, AcnEnsuranceErrorType>((sink) => {
      const run = ensure(request, (event) => sink.single(event)).pipe(
        Effect.mapError(normalizeEnsuranceError),
        Effect.match({
          onFailure: sink.fail,
          onSuccess: (instance) => {
            sink.single({ _tag: "Ready", instance })
            sink.end()
          },
        }),
      )
      return Effect.forkScoped(run)
    }, { bufferSize: "unbounded" })

  return AcnEnsurer.of({ ensure: ensureStream })
})
