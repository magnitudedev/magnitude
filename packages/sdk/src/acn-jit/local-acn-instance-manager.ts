import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { AcnReady, type AcnInstance, type AcnTarget } from "@magnitudedev/acn-protocol"
import {
  ProcessGroupController,
  ProcessGroupControllerLive,
  makeAcnOwnerStore,
  SqliteDriver,
} from "@magnitudedev/acn-protocol/coordination"
import { Effect, Scope, Stream } from "effect"
import { defaultDataDir } from "../binary"
import { AcnInstanceManager, type AcnEnsureEvent } from "./acn-instance-manager"
import { makeAcnCandidateLaunchSupervisor } from "./acn-candidate-launch-supervisor"
import { makeAcnDaemonShutdownSupervisor } from "./acn-daemon-shutdown-supervisor"
import { makeAcnEnsuranceCoordinator } from "./acn-ensurance-coordinator"
import { makeAcnDaemonLaunchCommandResolver, type AcnLaunchOverride } from "./acn-daemon-launch-command-resolver"
import { makeAcnOwnerObserver } from "./acn-owner-observer"
import { ChildProcessSpawner } from "./child-process"
import { AcnAdministrationFailed, type AcnEnsuranceError } from "./errors"

export type { AcnLaunchOverride } from "./acn-daemon-launch-command-resolver"

export interface LocalAcnInstanceManagerOptions {
  readonly binaryPath?: string
  readonly dataDir?: string
  readonly debug?: boolean
  readonly launchOverride?: AcnLaunchOverride
}

type ReadyInstance = AcnInstance<AcnReady>

/** @internal Composition root used by the live wrapper and deterministic tests. */
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
  | ProcessGroupController
  | Scope.Scope
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const http = yield* HttpClient.HttpClient
  const commandExecutor = yield* CommandExecutor.CommandExecutor
  const path = yield* Path.Path
  const spawner = yield* ChildProcessSpawner
  const processes = yield* ProcessGroupController
  const dataDirectory = options.dataDir ?? defaultDataDir()
  const owners = yield* makeAcnOwnerStore(dataDirectory).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.provideService(Path.Path, path),
  )

  const ownerObserver = makeAcnOwnerObserver(owners, processes, http)
  const shutdownSupervisor = yield* makeAcnDaemonShutdownSupervisor(owners, processes, http)
  const launchCommandResolver = makeAcnDaemonLaunchCommandResolver({
    binaryPath: options.binaryPath,
    dataDirectory,
    launchOverride: options.launchOverride,
    fileSystem: fs,
    httpClient: http,
    commandExecutor,
    path,
  })

  const ensureEffect = (
    target: AcnTarget,
    emit: (event: AcnEnsureEvent) => void,
  ): Effect.Effect<ReadyInstance, AcnEnsuranceError> => Effect.scoped(Effect.gen(function* () {
    const candidateSupervisor = yield* makeAcnCandidateLaunchSupervisor(spawner, processes)
    const coordinator = yield* makeAcnEnsuranceCoordinator({
      target,
      emit,
      debug: options.debug === true,
      dataDirectory,
      ownerObserver,
      shutdownSupervisor,
      candidateSupervisor,
      launchCommandResolver,
    })
    return yield* coordinator.run
  }))

  const ensure: AcnInstanceManager["ensure"] = (request) =>
    Stream.asyncPush<AcnEnsureEvent, AcnEnsuranceError>((sink) =>
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
    const observation = yield* ownerObserver.observe
    if (observation._tag === "AcnRecordedOwnerAbsent") return
    yield* shutdownSupervisor.shutdown(observation.owner, "AdministrativeStop")
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
  Effect.provideService(ProcessGroupController, ProcessGroupControllerLive),
)
