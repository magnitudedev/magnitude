import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as Path from "@effect/platform/Path"
import {
  type AcnIdentity,
  AcnHealthResponseSchema,
  type AcnHealthResponse,
  type AcnInstance,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol"
import { canUseAcnIdentity, compareAcnIdentities } from "@magnitudedev/acn-protocol/acn-identity"
import {
  AcnProcessStateConflict,
  applyAcnProcessCommand,
  currentProcessStartIdentity,
  readAcnProcessState,
  readProcessStartIdentity,
  type AcnProcessCommand,
  type AcnProcessRevision,
  type AcnProcessState,
  type AssignedAcn,
  type ExactAcnCandidate,
  type ExactIcnProcess,
  type ExactProcess,
} from "@magnitudedev/acn-protocol/process-state"
import type { ArtifactInstallationEvent } from "@magnitudedev/release"
import { Array as Arr, Clock, Effect, Option, Schema, Scope, Stream } from "effect"
import { defaultDataDir, resolveBinaryCommand } from "../binary"
import type { BinaryAcquisitionEvent } from "../binary"
import {
  AcnProcessManager,
  type AcnLaunchEvent,
  type AcnLaunchRequest,
} from "./acn-process-manager"
import { ChildProcessSpawner } from "./child-process"
import {
  DaemonDiscoveryFailed,
  DaemonError,
  DaemonSpawnFailed,
} from "./errors"
import {
  acnLifecycleObservationFromHealthState,
  acnStartupProgressKey,
} from "./lifecycle"

export interface LocalAcnProcessManagerOptions {
  readonly binaryPath?: string
  readonly dataDir?: string
  readonly debug?: boolean
  readonly probeTimeoutMs?: number
}

export type LocalAcnProcessManager = AcnProcessManager

const normalizeDaemonError = (error: unknown): DaemonError =>
  Schema.is(DaemonError)(error)
    ? error
    : new DaemonSpawnFailed({ reason: String(error) })

const sameProcess = (left: ExactProcess, right: ExactProcess): boolean =>
  left.pid === right.pid && left.processStartIdentity === right.processStartIdentity

const sameAcnOccurrence = (
  left: Pick<AssignedAcn, "id" | "pid" | "processStartIdentity">,
  right: Pick<AssignedAcn, "id" | "pid" | "processStartIdentity">,
): boolean => left.id === right.id && sameProcess(left, right)

const stateRevision = (state: Option.Option<AcnProcessState>): Option.Option<AcnProcessRevision> =>
  Option.map(state, (value) => value.revision)

const processIsExact = (process: ExactProcess) =>
  readProcessStartIdentity(process.pid).pipe(
    Effect.map(Option.exists((identity) => identity === process.processStartIdentity)),
  )

const waitForExactExit = (process: ExactProcess, durationMs: number) =>
  Effect.gen(function* () {
    const deadline = (yield* Clock.currentTimeMillis) + durationMs
    while ((yield* Clock.currentTimeMillis) < deadline) {
      if (!(yield* processIsExact(process))) return true
      yield* Effect.sleep("50 millis")
    }
    return !(yield* processIsExact(process))
  })

const signalExact = (process: ExactProcess, signal: NodeJS.Signals) =>
  Effect.gen(function* () {
    if (!(yield* processIsExact(process))) return
    yield* Effect.try({
      try: () => globalThis.process.kill(process.pid, signal),
      catch: (error) => new DaemonSpawnFailed({
        reason: `Failed to send ${signal} to exact process ${process.pid}: ${String(error)}`,
      }),
    }).pipe(Effect.catchIf(
      (error) => error.reason.includes("ESRCH"),
      () => Effect.void,
    ))
  })

const probeHealth = (
  current: AssignedAcn,
  timeoutMs: number,
  client: HttpClient.HttpClient,
) => client.execute(HttpClientRequest.get(`${current.url}/health`)).pipe(
  Effect.timeout(`${timeoutMs} millis`),
  Effect.flatMap((response) => response.json),
  Effect.flatMap(Schema.decodeUnknown(AcnHealthResponseSchema)),
  Effect.mapError((error) => new DaemonDiscoveryFailed({
    reason: `Failed to observe ACN ${current.id}: ${String(error)}`,
  })),
  Effect.filterOrFail(
    (health) =>
      health.id === current.id &&
      health.version === current.identity &&
      health.pid === current.pid,
    () => new DaemonDiscoveryFailed({
      reason: `ACN ${current.id} health does not match assigned process state`,
    }),
  ),
)

const instanceFrom = (current: AssignedAcn, health: AcnHealthResponse): AcnInstance => ({
  id: current.id,
  identity: current.identity,
  url: current.url,
  pid: current.pid,
  processStartIdentity: current.processStartIdentity,
  lifecycle: health.state,
})

const assignedIn = (state: AcnProcessState): Option.Option<AssignedAcn> => {
  if (state.mode._tag === "Assigned") return Option.some(state.mode.current)
  if (
    state.mode._tag === "Changing" &&
    state.mode.owner._tag === "Manager" &&
    state.mode.owner.phase._tag === "RetiringAssigned"
  ) return Option.some(state.mode.owner.phase.current)
  return Option.none()
}

const artifactProgress = (
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>,
): AcnStartupProgress => ({
  completed: event.progress.acceptedBytes,
  totalBytes: event.progress.totalBytes,
  unit: "Bytes",
  attempt: Option.some(event.progress.attempt),
})

const emitHealth = (health: AcnHealthResponse, emit: (event: AcnLaunchEvent) => void) =>
  Option.match(acnLifecycleObservationFromHealthState(health.state), {
    onNone: () => Effect.void,
    onSome: (observation) => Effect.sync(() => emit({ _tag: "Observation", observation })),
  })

interface PreparedCommand {
  readonly identity: AcnIdentity
  readonly command: readonly string[]
}

type ChangeObservation =
  | { readonly _tag: "Completed" }
  | { readonly _tag: "Failed"; readonly reason: string }
  | { readonly _tag: "Stalled"; readonly state: AcnProcessState }

type ReadinessObservation =
  | { readonly _tag: "Ready"; readonly instance: AcnInstance }
  | { readonly _tag: "Replace"; readonly current: AssignedAcn }

/** Local implementation of the exact-current ACN process contract. */
export const makeLocalAcnProcessManager = (
  options: LocalAcnProcessManagerOptions = {},
): Effect.Effect<
  LocalAcnProcessManager,
  never,
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | CommandExecutor.CommandExecutor
  | Path.Path
  | ChildProcessSpawner
  | Scope.Scope
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const client = yield* HttpClient.HttpClient
  const commandExecutor = yield* CommandExecutor.CommandExecutor
  const path = yield* Path.Path
  const spawner = yield* ChildProcessSpawner
  const dataDirectory = options.dataDir ?? defaultDataDir()
  const probeTimeoutMs = options.probeTimeoutMs ?? 2_000
  const manager: ExactProcess = {
    pid: process.pid,
    processStartIdentity: yield* currentProcessStartIdentity.pipe(Effect.orDie),
  }
  const operationScope = yield* Effect.scope
  const reconcilePermit = yield* Effect.makeSemaphore(1)

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

  const observeAssigned = (current: AssignedAcn) => Effect.gen(function* () {
    if (!(yield* provideLocal(processIsExact(current)))) return Option.none<AcnInstance>()
    const health = yield* probeHealth(current, probeTimeoutMs, client)
    return Option.some(instanceFrom(current, health))
  })

  const observeCurrent = Effect.gen(function* () {
    const state = yield* readState
    if (Option.isNone(state)) return Option.none<AcnInstance>()
    return yield* Option.match(assignedIn(state.value), {
      onNone: () => Effect.succeed(Option.none<AcnInstance>()),
      onSome: observeAssigned,
    })
  })

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
    process: ExactProcess,
    signal: NodeJS.Signals,
  ) => Effect.gen(function* () {
    if (!(yield* stillOwnsRevision(revision))) return false
    yield* provideLocal(signalExact(process, signal))
    return true
  })

  const stopExact = (
    state: AcnProcessState,
    process: ExactProcess,
    requestShutdown: Effect.Effect<unknown, unknown> = Effect.void,
  ) => Effect.gen(function* () {
    if (!(yield* stillOwnsRevision(state.revision))) return false
    yield* requestShutdown.pipe(Effect.timeout("750 millis"), Effect.ignore)
    if (yield* provideLocal(waitForExactExit(process, 2_000))) return true
    if (!(yield* signalWhileOwned(state.revision, process, "SIGINT"))) return false
    if (yield* provideLocal(waitForExactExit(process, 2_000))) return true
    if (!(yield* signalWhileOwned(state.revision, process, "SIGKILL"))) return false
    return yield* provideLocal(waitForExactExit(process, 2_000))
  })

  const stopIcn = (state: AcnProcessState, icn: ExactIcnProcess) =>
    Effect.gen(function* () {
      if (!(yield* provideLocal(processIsExact(icn)))) return true
      if (!(yield* signalWhileOwned(state.revision, icn, "SIGTERM"))) return false
      if (yield* provideLocal(waitForExactExit(icn, 1_000))) return true
      if (!(yield* signalWhileOwned(state.revision, icn, "SIGKILL"))) return false
      return yield* provideLocal(waitForExactExit(icn, 1_000))
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

  const resolveCommand = (
    request: AcnLaunchRequest,
    identity: AcnIdentity,
    emit: (event: AcnLaunchEvent) => void,
  ): Effect.Effect<PreparedCommand, DaemonError> =>
    Option.match(request.command, {
      onSome: (command) => identity === request.identity
        ? Effect.succeed({ identity, command: Array.from(command) })
        : Effect.fail(new DaemonSpawnFailed({
            reason: `Configured ACN command is for ${request.identity}, not adopted identity ${identity}`,
          })),
      onNone: () => {
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
      },
    }).pipe(Effect.mapError(normalizeDaemonError))

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
        onNone: () => Effect.fail(new DaemonSpawnFailed({ reason: `Spawned ACN ${pid} exited before publication` })),
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
  ) =>
    reconcilePermit.withPermits(1)(Effect.gen(function* () {
      const fresh = yield* readState
      if (Option.isNone(fresh) || fresh.value.revision !== state.revision) return
      const current = fresh.value
      if (
        current.mode._tag !== "Changing" ||
        current.mode.owner._tag !== "Manager" ||
        !sameProcess(current.mode.owner.process, manager)
      ) return
      switch (current.mode.owner.phase._tag) {
        case "RetiringAssigned": {
          const retired = yield* retireAssigned(current, current.mode.owner.phase.current)
          yield* apply(Option.some(current), retired
            ? { _tag: "PredecessorExited", manager }
            : {
                _tag: "RetirementBlocked",
                manager,
                reason: `Could not prove ACN ${current.mode.owner.phase.current.id} and its ICN absent`,
              }).pipe(Effect.catchTag("AcnProcessStateConflict", () => Effect.void))
          return
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
          return
        }
        case "BlockedCandidateCleanup":
          return
        case "Spawning": {
          if (Option.isNone(prepared)) return
          yield* spawnCandidate(current, prepared.value)
        }
      }
    }))

  const reconcileChange = (
    initial: AcnProcessState,
    prepared: Option.Option<PreparedCommand>,
  ) => Effect.gen(function* () {
    let observedRevision = initial.revision
    let unchangedSince = yield* Clock.currentTimeMillis
    while (true) {
      const stateOption = yield* readState
      if (Option.isNone(stateOption)) return
      const state = stateOption.value
      if (state.mode._tag !== "Changing") return
      if (state.revision !== observedRevision) {
        observedRevision = state.revision
        unchangedSince = yield* Clock.currentTimeMillis
      }
      if (state.mode.owner._tag === "Manager" && sameProcess(state.mode.owner.process, manager)) {
        if (state.mode.owner.phase._tag === "BlockedCandidateCleanup") return
        if (
          state.mode.owner.phase._tag === "Spawning" &&
          (
            Option.isNone(prepared) ||
            state.mode.purpose._tag !== "Ensure" ||
            prepared.value.identity !== state.mode.purpose.target
          )
        ) return
        yield* reconcileOwned(state, prepared)
        yield* Effect.sleep("50 millis")
        continue
      }
      const owner = state.mode.owner._tag === "Manager"
        ? state.mode.owner.process
        : state.mode.owner.candidate
      const alive = yield* provideLocal(processIsExact(owner))
      if (!alive && state.mode.owner._tag === "Candidate") {
        yield* apply(Option.some(state), {
          _tag: "CandidateFailed",
          candidate: state.mode.owner.candidate,
          reason: `Candidate ACN ${state.mode.owner.candidate.pid} exited before admission`,
        }).pipe(Effect.catchTag("AcnProcessStateConflict", () => Effect.void))
        return
      }
      const canContinue = state.mode.purpose._tag === "Terminate" || (
        Option.isSome(prepared) && prepared.value.identity === state.mode.purpose.target
      )
      if (canContinue && (!alive || (yield* Clock.currentTimeMillis) - unchangedSince >= 30_000)) {
        yield* apply(Option.some(state), { _tag: "TakeOver", manager }).pipe(
          Effect.catchTag("AcnProcessStateConflict", () => Effect.void),
        )
        yield* Effect.sleep("50 millis")
        continue
      }
      yield* Effect.sleep("100 millis")
    }
  })

  const applyAndReconcile = (
    state: Option.Option<AcnProcessState>,
    command: AcnProcessCommand,
    prepared: Option.Option<PreparedCommand>,
  ) => Effect.uninterruptible(
    Effect.gen(function* () {
      const changed = yield* apply(state, command)
      yield* Effect.forkIn(reconcileChange(changed, prepared), operationScope)
      return changed
    }),
  )

  const observeChange = (initial: AcnProcessState): Effect.Effect<ChangeObservation, DaemonError> =>
    Effect.gen(function* () {
      if (initial.mode._tag !== "Changing") return { _tag: "Completed" } as const
      const changeRevision = initial.mode.changeRevision
      let observedRevision = initial.revision
      let unchangedSince = yield* Clock.currentTimeMillis
      while (true) {
        const observed = yield* readState
        if (Option.isNone(observed)) return { _tag: "Completed" } as const
        const state = observed.value
        if (state.mode._tag !== "Changing") {
          const result = Option.getOrUndefined(state.mode.result)
          return result?._tag === "Failed" && result.changeRevision === changeRevision
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
        if (!alive || (yield* Clock.currentTimeMillis) - unchangedSince >= 30_000) {
          return { _tag: "Stalled", state } as const
        }
        yield* Effect.sleep("100 millis")
      }
    }).pipe(Effect.mapError(normalizeDaemonError))

  const waitUntilReady = (
    targetIdentity: string,
    emit: (event: AcnLaunchEvent) => void,
  ): Effect.Effect<ReadinessObservation, DaemonError> => Effect.gen(function* () {
    let progressKey: string | undefined
    let progressDeadline = (yield* Clock.currentTimeMillis) + 30_000
    while (true) {
      const state = yield* readState
      if (Option.isNone(state)) return yield* new DaemonSpawnFailed({ reason: "ACN assignment disappeared" })
      if (state.value.mode._tag === "Changing") {
        yield* Effect.sleep("100 millis")
        continue
      }
      if (state.value.mode._tag === "Unassigned") {
        const result = Option.getOrUndefined(state.value.mode.result)
        return yield* new DaemonSpawnFailed({
          reason: result?._tag === "Failed" ? result.reason : "ACN launch ended without assignment",
        })
      }
      const current = state.value.mode.current
      if (!canUseAcnIdentity(targetIdentity, current.identity)) {
        return yield* new DaemonSpawnFailed({ reason: `Assigned ACN ${current.identity} does not satisfy ${targetIdentity}` })
      }
      if (!(yield* provideLocal(processIsExact(current)))) {
        return { _tag: "Replace", current } as const
      }
      const health = yield* probeHealth(current, probeTimeoutMs, client)
      yield* emitHealth(health, emit)
      const instance = instanceFrom(current, health)
      if (health.state._tag === "Ready") return { _tag: "Ready", instance } as const
      if (health.state._tag === "Stopping") {
        return { _tag: "Replace", current } as const
      }
      const nextKey = acnStartupProgressKey(health.state)
      if (nextKey !== progressKey) {
        progressKey = nextKey
        progressDeadline = (yield* Clock.currentTimeMillis) + 30_000
      } else if ((yield* Clock.currentTimeMillis) >= progressDeadline) {
        return { _tag: "Replace", current } as const
      }
      yield* Effect.sleep("100 millis")
    }
  }).pipe(Effect.mapError(normalizeDaemonError))

  const targetFor = (
    requestIdentity: AcnIdentity,
    state: Option.Option<AcnProcessState>,
  ): AcnIdentity => Option.match(state, {
    onNone: () => requestIdentity,
    onSome: (current) => compareAcnIdentities(requestIdentity, current.identityFloor) >= 0
      ? requestIdentity
      : current.identityFloor,
  })

  const ensure = (request: AcnLaunchRequest, emit: (event: AcnLaunchEvent) => void) =>
    Effect.gen(function* () {
      let forced = Option.map(request.replace, (instance) => ({
        id: instance.id,
        pid: instance.pid,
        processStartIdentity: instance.processStartIdentity,
      }))
      let stalledRevision = Option.none<AcnProcessRevision>()

      while (true) {
        let state = yield* readState
        let targetIdentity = targetFor(request.identity, state)

        if (Option.isSome(state) && state.value.mode._tag === "Assigned") {
          const current = state.value.mode.current
          const forceCurrent = Option.exists(forced, (replace) => sameAcnOccurrence(replace, current))
          if (!forceCurrent && canUseAcnIdentity(targetIdentity, current.identity)) {
            const readiness = yield* waitUntilReady(targetIdentity, emit)
            if (readiness._tag === "Ready") return readiness.instance
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
                Option.none(),
              ).pipe(Effect.either)
              if (taken._tag === "Left" && !(taken.left instanceof AcnProcessStateConflict)) {
                return yield* taken.left
              }
              stalledRevision = Option.none()
              continue
            }
            const observation = yield* observeChange(changing)
            if (observation._tag === "Failed") {
              return yield* new DaemonSpawnFailed({ reason: observation.reason })
            }
            stalledRevision = observation._tag === "Stalled"
              ? Option.some(observation.state.revision)
              : Option.none()
            continue
          }

          const sufficient = canUseAcnIdentity(targetIdentity, mode.purpose.target)
          const blocked = mode.owner._tag === "Manager" &&
            mode.owner.phase._tag === "BlockedCandidateCleanup"
          if (sufficient && !blocked && !Option.contains(stalledRevision, changing.revision)) {
            const observation = yield* observeChange(changing)
            if (observation._tag === "Failed") {
              return yield* new DaemonSpawnFailed({ reason: observation.reason })
            }
            stalledRevision = observation._tag === "Stalled"
              ? Option.some(observation.state.revision)
              : Option.none()
            continue
          }
        }

        const prepared = yield* resolveCommand(request, targetIdentity, emit)
        state = yield* readState
        targetIdentity = targetFor(request.identity, state)
        if (prepared.identity !== targetIdentity) {
          stalledRevision = Option.none()
          continue
        }

        let transition: AcnProcessCommand | undefined
        if (Option.isNone(state) || state.value.mode._tag === "Unassigned") {
          transition = { _tag: "BeginEnsure", target: targetIdentity, manager }
        } else if (state.value.mode._tag === "Assigned") {
          const current = state.value.mode.current
          const forceCurrent = Option.exists(forced, (replace) => sameAcnOccurrence(replace, current))
          if (!forceCurrent && canUseAcnIdentity(targetIdentity, current.identity)) continue
          transition = { _tag: "BeginReplacement", target: targetIdentity, manager, current }
        } else if (state.value.mode.purpose._tag === "Terminate") {
          stalledRevision = Option.none()
          continue
        } else if (!canUseAcnIdentity(targetIdentity, state.value.mode.purpose.target)) {
          transition = { _tag: "UpgradeEnsure", target: targetIdentity, manager }
        } else if (
          state.value.mode.owner._tag === "Manager" &&
          state.value.mode.owner.phase._tag === "BlockedCandidateCleanup"
        ) {
          transition = { _tag: "RetryCandidateCleanup", manager }
        } else if (Option.contains(stalledRevision, state.value.revision)) {
          transition = { _tag: "TakeOver", manager }
        } else {
          stalledRevision = Option.none()
          continue
        }

        const changed = yield* applyAndReconcile(
          state,
          transition,
          Option.some(prepared),
        ).pipe(Effect.either)
        if (changed._tag === "Left") {
          if (changed.left instanceof AcnProcessStateConflict) continue
          return yield* changed.left
        }
        stalledRevision = Option.none()
        const observation = yield* observeChange(changed.right)
        if (observation._tag === "Failed") {
          return yield* new DaemonSpawnFailed({ reason: observation.reason })
        }
        if (observation._tag === "Stalled") {
          stalledRevision = Option.some(observation.state.revision)
        }
      }
    }).pipe(Effect.mapError(normalizeDaemonError))

  const launch: AcnProcessManager["launch"] = (request) =>
    Stream.asyncPush<AcnLaunchEvent, DaemonError>((sink) => {
      const run = ensure(request, (event) => sink.single(event)).pipe(
        Effect.mapError(normalizeDaemonError),
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

  const terminate = (instance: AcnInstance) => Effect.gen(function* () {
    let state = yield* readState
    if (Option.isNone(state) || state.value.mode._tag !== "Assigned") return
    const current = state.value.mode.current
    if (current.id !== instance.id || !sameProcess(current, instance)) {
      return yield* new DaemonSpawnFailed({ reason: `ACN ${instance.id} is no longer the assigned exact process` })
    }
    const changed = yield* applyAndReconcile(
      state,
      { _tag: "BeginTerminate", manager, current },
      Option.none(),
    )
    const result = yield* observeChange(changed)
    if (result._tag === "Failed") {
      return yield* new DaemonSpawnFailed({ reason: result.reason })
    }
    if (result._tag === "Stalled") {
      return yield* new DaemonSpawnFailed({ reason: `ACN ${instance.id} termination stalled` })
    }
  })

  return AcnProcessManager.of({
    observeCurrent: observeCurrent.pipe(Effect.mapError(normalizeDaemonError)),
    launch,
    terminate: (instance) => terminate(instance).pipe(Effect.mapError(normalizeDaemonError)),
  })
})
