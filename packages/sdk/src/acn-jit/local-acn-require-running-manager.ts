import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { type AcnInstance, AcnReady, type AcnTarget } from "@magnitudedev/acn-protocol"
import {
  ProcessGroupController,
  SqliteDriver,
  makeAcnOwnerStore,
} from "@magnitudedev/acn-protocol/coordination"
import { ProcessGroupControllerLive } from "@magnitudedev/acn-protocol/coordination/exact-process"
import { Duration, Effect, Option, Stream } from "effect"
import { defaultDataDir } from "../binary"
import {
  ACN_ENSURE_TIMEOUT,
  AcnInstanceManager,
  type AcnEnsureEvent,
} from "./acn-instance-manager"
import { makeAcnOwnerObserver, type AcnOwnerObserver } from "./acn-owner-observer"
import {
  AcnAdministrationFailed,
  AcnEnsuranceFailed,
  AcnHealthUnavailable,
  type AcnEnsuranceError,
} from "./errors"
import { acnLifecycleObservationFromHealthState } from "./lifecycle"

type ReadyInstance = AcnInstance<AcnReady>

const unavailable = (detail?: string) => new AcnEnsuranceFailed({
  reason: detail === undefined
    ? "Magnitude service is not running. Run `magnitude service start`."
    : `${detail} Run \`magnitude service start\`.`,
})

const observeRunning = (
  target: AcnTarget,
  observer: AcnOwnerObserver,
  emit: (event: AcnEnsureEvent) => void,
  absentOwner: "Fail" | "Wait",
): Effect.Effect<ReadyInstance, AcnEnsuranceError> => Effect.gen(function* () {
  while (true) {
    const observation = yield* observer.observe
    switch (observation._tag) {
      case "AcnRecordedOwnerAbsent":
        if (absentOwner === "Wait") {
          yield* Effect.sleep(Duration.millis(250))
          continue
        }
        return yield* unavailable()
      case "AcnRecordedOwnerProcessGroupSurvives":
        return yield* unavailable("The recorded Magnitude service process is unavailable.")
      case "AcnRecordedOwnerLiveWithoutHealth":
        if (absentOwner === "Wait") {
          yield* Effect.sleep(Duration.millis(250))
          continue
        }
        return yield* new AcnHealthUnavailable({
          owner: observation.owner,
          attempts: observation.attempts,
        })
      case "AcnRecordedOwnerLiveWithHealth": {
        const { health } = observation.health
        if (health.revision < target.revision) {
          return yield* unavailable("The running Magnitude service is incompatible with this client.")
        }
        if (health.state._tag === "Stopping") {
          return yield* unavailable("The Magnitude service is stopping.")
        }
        if (health.state._tag === "Starting") {
          const progress = acnLifecycleObservationFromHealthState(health.state)
          if (Option.isSome(progress)) {
            emit({ _tag: "Observation", observation: progress.value })
          }
          yield* Effect.sleep(Duration.millis(250))
          continue
        }
        const ready = yield* observer.confirmReady(observation.owner, observation.health)
        if (Option.isSome(ready)) return ready.value
        yield* Effect.sleep(Duration.millis(250))
      }
    }
  }
}).pipe(Effect.timeoutFail({
  duration: ACN_ENSURE_TIMEOUT,
  onTimeout: () => unavailable("The running Magnitude service did not become ready."),
}))

const makeObservedAcnInstanceManager = (
  observer: AcnOwnerObserver,
  absentOwner: "Fail" | "Wait",
): AcnInstanceManager => {
  const ensure: AcnInstanceManager["ensure"] = (request) =>
    Stream.asyncPush<AcnEnsureEvent, AcnEnsuranceError>((sink) =>
      Effect.forkScoped(observeRunning(
        request.target,
        observer,
        (event) => sink.single(event),
        absentOwner,
      ).pipe(Effect.match({
        onFailure: sink.fail,
        onSuccess: (instance) => {
          sink.single({ _tag: "Ready", instance })
          sink.end()
        },
      }))), { bufferSize: "unbounded" })
  const stop = Effect.fail(new AcnAdministrationFailed({
    reason: "This client is not authorized to stop the Magnitude service",
  }))
  return AcnInstanceManager.of({ ensure, stop })
}

/** @internal Deterministic composition point used by tests. */
export const makeRequireRunningAcnInstanceManagerFromObserver = (
  observer: AcnOwnerObserver,
): AcnInstanceManager => makeObservedAcnInstanceManager(observer, "Fail")

/** @internal Deterministic composition point used by tests. */
export const makeStartingAcnInstanceManagerFromObserver = (
  observer: AcnOwnerObserver,
): AcnInstanceManager => makeObservedAcnInstanceManager(observer, "Wait")

export interface LocalAcnObservationOptions {
  readonly dataDir?: string
}

const makeLocalObservedAcnInstanceManager = (
  options: LocalAcnObservationOptions,
  absentOwner: "Fail" | "Wait",
): Effect.Effect<
  AcnInstanceManager,
  never,
  | FileSystem.FileSystem
  | HttpClient.HttpClient
  | Path.Path
  | SqliteDriver
> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const processes = yield* ProcessGroupController
  const http = yield* HttpClient.HttpClient
  const owners = yield* makeAcnOwnerStore(options.dataDir ?? defaultDataDir()).pipe(
    Effect.provideService(FileSystem.FileSystem, fs),
    Effect.provideService(Path.Path, path),
  )
  return makeObservedAcnInstanceManager(
    makeAcnOwnerObserver(owners, processes, http),
    absentOwner,
  )
}).pipe(Effect.provideService(ProcessGroupController, ProcessGroupControllerLive))

/** Observes an existing service and fails immediately when it is unavailable. */
export const makeLocalAcnRequireRunningInstanceManager = (
  options: LocalAcnObservationOptions = {},
) => makeLocalObservedAcnInstanceManager(options, "Fail")

/** Follows the persistent service explicitly launched by `magnitude service start`. */
export const makeLocalAcnStartingInstanceManager = (
  options: LocalAcnObservationOptions = {},
) => makeLocalObservedAcnInstanceManager(options, "Wait")
