import {
  AcnReady,
  type AcnHealthState,
  type AcnInstance,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"
import { sameAcnOwner, type AcnOwnerRecord } from "@magnitudedev/acn-protocol/coordination"
import { Array as Arr, Clock, Context, Duration, Effect, Option, Ref } from "effect"
import { ACN_ENSURE_TIMEOUT, type AcnEnsureEvent } from "./acn-instance-manager"
import type { AcnCandidateLaunchState, AcnCandidateLaunchSupervisor } from "./acn-candidate-launch-supervisor"
import type { AcnDaemonShutdownSupervisor } from "./acn-daemon-shutdown-supervisor"
import { AcnConvergenceDecider } from "./acn-convergence-decider"
import type { AcnDaemonLaunchCommandResolver, AcnDaemonLaunchCommand } from "./acn-daemon-launch-command-resolver"
import type { AcnOwnerObservation, AcnOwnerObserver } from "./acn-owner-observer"
import {
  AcnCandidateAdmissionLost,
  AcnCandidateAdmissionTimedOut,
  AcnCandidateExitedAfterAdmissionFailure,
  AcnCandidateExitedBeforeAdmissionFailure,
  AcnCandidateOwnershipLostAfterAdmission,
  AcnDaemonStartupTimedOut,
  AcnEnsuranceConvergenceTimedOut,
  type AcnEnsuranceError,
} from "./errors"
import { acnLifecycleObservationFromHealthState } from "./lifecycle"

type ReadyInstance = AcnInstance<AcnReady>

type ObservationContinuity =
  | { readonly _tag: "Initial" }
  | { readonly _tag: "OwnerAbsent" }
  | { readonly _tag: "OwnerProcessGroupSurvives" }
  | { readonly _tag: "HealthUnavailable" }
  | { readonly _tag: "HealthState"; readonly state: AcnHealthState["_tag"] }

interface ConvergenceMemory {
  readonly owner: Option.Option<AcnOwnerRecord>
  readonly continuity: ObservationContinuity
  readonly ownerObservedAt: number
  readonly healthStateObservedAt: number
}

const RECONCILIATION_INTERVAL = Duration.seconds(1)
const monotonicMillis = Clock.currentTimeNanos.pipe(
  Effect.map((nanos) => Number(nanos / 1_000_000n)),
)

const observationOwner = (observation: AcnOwnerObservation): Option.Option<AcnOwnerRecord> => {
  switch (observation._tag) {
    case "AcnRecordedOwnerAbsent": return observation.expectedOwner
    case "AcnRecordedOwnerProcessGroupSurvives": return Option.some(observation.owner)
    case "AcnRecordedOwnerLiveWithoutHealth": return Option.some(observation.owner)
    case "AcnRecordedOwnerLiveWithHealth": return Option.some(observation.owner)
  }
}

const observationContinuity = (observation: AcnOwnerObservation): ObservationContinuity => {
  switch (observation._tag) {
    case "AcnRecordedOwnerAbsent": return { _tag: "OwnerAbsent" }
    case "AcnRecordedOwnerProcessGroupSurvives": return { _tag: "OwnerProcessGroupSurvives" }
    case "AcnRecordedOwnerLiveWithoutHealth": return { _tag: "HealthUnavailable" }
    case "AcnRecordedOwnerLiveWithHealth": return {
      _tag: "HealthState",
      state: observation.health.health.state._tag,
    }
  }
}

const sameContinuity = (left: ObservationContinuity, right: ObservationContinuity): boolean =>
  left._tag === right._tag &&
  (left._tag !== "HealthState" || (right._tag === "HealthState" && left.state === right.state))

const sameOptionalOwner = (
  left: Option.Option<AcnOwnerRecord>,
  right: Option.Option<AcnOwnerRecord>,
): boolean => Option.match(left, {
  onNone: () => Option.isNone(right),
  onSome: (owner) => Option.exists(right, (candidate) => sameAcnOwner(owner, candidate)),
})

const updateMemory = (
  previous: ConvergenceMemory,
  observation: AcnOwnerObservation,
  now: number,
): ConvergenceMemory => {
  const owner = observationOwner(observation)
  const continuity = observationContinuity(observation)
  const ownerChanged = !sameOptionalOwner(previous.owner, owner)
  return {
    owner,
    continuity,
    ownerObservedAt: ownerChanged ? now : previous.ownerObservedAt,
    healthStateObservedAt: ownerChanged || !sameContinuity(previous.continuity, continuity)
      ? now
      : previous.healthStateObservedAt,
  }
}

export interface AcnEnsuranceCoordinator {
  readonly run: Effect.Effect<ReadyInstance, AcnEnsuranceError>
}
export const AcnEnsuranceCoordinator = Context.GenericTag<AcnEnsuranceCoordinator>(
  "@magnitudedev/sdk/AcnEnsuranceCoordinator",
)

export interface MakeAcnEnsuranceCoordinatorOptions {
  readonly target: AcnTarget
  readonly emit: (event: AcnEnsureEvent) => void
  readonly debug: boolean
  readonly dataDirectory: string
  readonly ownerObserver: AcnOwnerObserver
  readonly shutdownSupervisor: AcnDaemonShutdownSupervisor
  readonly candidateSupervisor: AcnCandidateLaunchSupervisor
  readonly launchCommandResolver: AcnDaemonLaunchCommandResolver
}

export const makeAcnEnsuranceCoordinator = (
  options: MakeAcnEnsuranceCoordinatorOptions,
): Effect.Effect<AcnEnsuranceCoordinator> => Effect.gen(function* () {
  const prepared = yield* Ref.make(Option.none<AcnDaemonLaunchCommand>())
  const startedAt = yield* monotonicMillis
  const memory = yield* Ref.make<ConvergenceMemory>({
    owner: Option.none(),
    continuity: { _tag: "Initial" },
    ownerObservedAt: startedAt,
    healthStateObservedAt: startedAt,
  })

  const prepare = Effect.gen(function* () {
    const existing = yield* Ref.get(prepared)
    if (Option.isSome(existing)) return existing.value
    const value = yield* options.launchCommandResolver.resolve(
      options.target,
      (event) => Effect.sync(() => options.emit(event)),
    )
    yield* Ref.set(prepared, Option.some(value))
    return value
  })

  const coordinate = Effect.gen(function* () {
    while (true) {
      const now = yield* monotonicMillis
      const observation = yield* options.ownerObserver.observe
      const currentMemory = yield* Ref.modify(memory, (previous) => {
        const next = updateMemory(previous, observation, now)
        return [next, next]
      })
      const candidate = yield* options.candidateSupervisor.reconcile(observationOwner(observation))

      if (observation._tag === "AcnRecordedOwnerLiveWithHealth") {
        const progress = acnLifecycleObservationFromHealthState(observation.health.health.state)
        if (Option.isSome(progress)) {
          yield* Effect.sync(() => options.emit({ _tag: "Observation", observation: progress.value }))
        }
      }

      const decision = AcnConvergenceDecider.decide({
        target: options.target,
        observation,
        candidate,
        launchPrepared: Option.isSome(yield* Ref.get(prepared)),
        now,
        ownerObservedAt: currentMemory.ownerObservedAt,
        healthStateObservedAt: currentMemory.healthStateObservedAt,
      })

      switch (decision._tag) {
        case "Wait":
          yield* Effect.sleep(RECONCILIATION_INTERVAL)
          break
        case "PrepareLaunch":
          yield* prepare
          break
        case "LaunchCandidate": {
          const launch = yield* prepare
          const command: Arr.NonEmptyReadonlyArray<string> = [
            ...launch.command,
            ...(options.debug && !launch.command.includes("--debug") ? ["--debug"] : []),
            "--parent-bound", "--data-dir", options.dataDirectory,
          ]
          yield* options.candidateSupervisor.launch(command)
          break
        }
        case "ShutdownDaemon": {
          yield* options.shutdownSupervisor.shutdown(decision.owner, decision.reason)
          break
        }
        case "ShutdownDaemonThenFail": {
          yield* options.shutdownSupervisor.shutdown(decision.owner, decision.reason)
          return yield* new AcnDaemonStartupTimedOut({ owner: decision.owner })
        }
        case "ConfirmReady": {
          const ready = yield* options.ownerObserver.confirmReady(decision.owner, decision.observed)
          if (Option.isNone(ready)) break
          if (candidate._tag === "Admitted") yield* options.candidateSupervisor.markReady(ready.value)
          return ready.value
        }
        case "FailCandidateLaunch":
          return yield* decision.failure
        case "FailCandidateExitedBeforeAdmission":
          return yield* new AcnCandidateExitedBeforeAdmissionFailure(decision)
        case "FailCandidateExitedAfterAdmission":
          return yield* new AcnCandidateExitedAfterAdmissionFailure(decision)
        case "FailCandidateAdmissionTimedOut":
          return yield* new AcnCandidateAdmissionTimedOut({ process: decision.process })
        case "FailCandidateAdmissionLost":
          return yield* new AcnCandidateAdmissionLost({ process: decision.process })
        case "FailCandidateOwnershipLostAfterAdmission":
          return yield* new AcnCandidateOwnershipLostAfterAdmission({ process: decision.process })
      }
    }
  })

  return AcnEnsuranceCoordinator.of({
    run: coordinate.pipe(Effect.timeoutFail({
      duration: ACN_ENSURE_TIMEOUT,
      onTimeout: () => new AcnEnsuranceConvergenceTimedOut({}),
    })),
  })
})
