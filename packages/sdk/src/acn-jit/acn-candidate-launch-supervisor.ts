import { FSM } from "@magnitudedev/utils"
import { AcnReady, type AcnInstance } from "@magnitudedev/acn-protocol"
import {
  ExactProcessSchema,
  ExactProcessIdentityObservationFailed,
  ProcessGroupController,
  type AcnOwnerRecord,
  type ExactProcess,
} from "@magnitudedev/acn-protocol/coordination"
import {
  Array as Arr,
  Clock,
  Context,
  Duration,
  Effect,
  Option,
  Ref,
  Schema,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  type AcnCandidateCleanupError,
  type AcnCandidateExactProcessConfirmationError,
  type ChildProcessSpawner,
  type SpawnedAcnCandidate,
} from "./child-process"
import {
  AcnCandidateAdmissionAcknowledgementTimedOut,
  type AcnCandidateAdmissionAlreadyAcknowledged,
  type AcnCandidateAdmissionBeforeExactProcessConfirmed,
  AcnCandidateExitedBeforeIdentityObserved,
  AcnCandidateExactProcessAlreadyConfirmed,
  AcnCandidateExactProcessPidMismatch,
  AcnCandidateIdentityUnavailable,
  AcnCandidateLaunchAlreadyAttempted,
  type AcnCandidateParentChannelReleaseFailed,
  AcnCandidateReadyBeforeAdmission,
  AcnCandidateReadyInstanceMismatch,
  AcnCandidateSpawnFailed,
  AcnProcessIdentityObservationTimedOut,
} from "./errors"
import { inspectExactProcess } from "./acn-owner-observer"

const CANDIDATE_ADMISSION_TIMEOUT = Duration.seconds(30)
const CANDIDATE_PARENT_RELEASE_TIMEOUT = Duration.seconds(2)
const CANDIDATE_EXIT_DIAGNOSTIC_TIMEOUT = Duration.seconds(2)

export class AcnCandidateNotLaunched extends Schema.TaggedClass<AcnCandidateNotLaunched>()(
  "NotLaunched", {},
) {}
export class AcnCandidateSpawned extends Schema.TaggedClass<AcnCandidateSpawned>()(
  "Spawned", { process: ExactProcessSchema, launchedAt: Schema.Number },
) {}
export class AcnCandidateAdmitted extends Schema.TaggedClass<AcnCandidateAdmitted>()(
  "Admitted", { process: ExactProcessSchema, launchedAt: Schema.Number, admittedAt: Schema.Number },
) {}
export class AcnCandidateReady extends Schema.TaggedClass<AcnCandidateReady>()(
  "Ready", { process: ExactProcessSchema, launchedAt: Schema.Number, admittedAt: Schema.Number },
) {}
export const AcnCandidateLaunchFailureSchema = Schema.Union(
  AcnCandidateSpawnFailed,
  ExactProcessIdentityObservationFailed,
  AcnProcessIdentityObservationTimedOut,
  AcnCandidateIdentityUnavailable,
  AcnCandidateExitedBeforeIdentityObserved,
  AcnCandidateExactProcessPidMismatch,
  AcnCandidateExactProcessAlreadyConfirmed,
)
export type AcnCandidateLaunchFailure = typeof AcnCandidateLaunchFailureSchema.Type
export class AcnCandidateLaunchFailed extends Schema.TaggedClass<AcnCandidateLaunchFailed>()(
  "LaunchFailed", { failure: AcnCandidateLaunchFailureSchema },
) {}
const CandidateExitFields = {
  process: ExactProcessSchema,
  launchedAt: Schema.Number,
  code: Schema.Number,
  stderr: Schema.String,
}
export class AcnCandidateExitedBeforeAdmission extends Schema.TaggedClass<AcnCandidateExitedBeforeAdmission>()(
  "ExitedBeforeAdmission", CandidateExitFields,
) {}
export class AcnCandidateExitedAfterAdmission extends Schema.TaggedClass<AcnCandidateExitedAfterAdmission>()(
  "ExitedAfterAdmission", {
    ...CandidateExitFields,
    admittedAt: Schema.Number,
  },
) {}
export class AcnCandidateAdmissionAcknowledgementLost extends Schema.TaggedClass<AcnCandidateAdmissionAcknowledgementLost>()(
  "AdmissionAcknowledgementLost", {
    process: ExactProcessSchema,
    launchedAt: Schema.Number,
    lostAt: Schema.Number,
  },
) {}
export class AcnCandidateAdmissionExpired extends Schema.TaggedClass<AcnCandidateAdmissionExpired>()(
  "AdmissionExpired", { process: ExactProcessSchema, launchedAt: Schema.Number },
) {}
export class AcnCandidateLostAfterAdmission extends Schema.TaggedClass<AcnCandidateLostAfterAdmission>()(
  "LostAfterAdmission", {
    process: ExactProcessSchema,
    launchedAt: Schema.Number,
    admittedAt: Schema.Number,
    lostAt: Schema.Number,
  },
) {}

export const AcnCandidateLaunchStateSchema = Schema.Union(
  AcnCandidateNotLaunched,
  AcnCandidateSpawned,
  AcnCandidateAdmitted,
  AcnCandidateReady,
  AcnCandidateLaunchFailed,
  AcnCandidateExitedBeforeAdmission,
  AcnCandidateExitedAfterAdmission,
  AcnCandidateAdmissionAcknowledgementLost,
  AcnCandidateAdmissionExpired,
  AcnCandidateLostAfterAdmission,
)
export type AcnCandidateLaunchState = typeof AcnCandidateLaunchStateSchema.Type
export type AcnCandidateSupervisionError =
  | AcnCandidateSpawnFailed
  | AcnCandidateLaunchAlreadyAttempted
  | AcnCandidateIdentityUnavailable
  | AcnCandidateExitedBeforeIdentityObserved
  | AcnCandidateAdmissionAlreadyAcknowledged
  | AcnCandidateAdmissionBeforeExactProcessConfirmed
  | AcnCandidateParentChannelReleaseFailed
  | AcnCandidateAdmissionAcknowledgementTimedOut
  | AcnCandidateReadyBeforeAdmission
  | AcnCandidateReadyInstanceMismatch
  | AcnCandidateCleanupError
  | AcnCandidateExactProcessConfirmationError
  | ExactProcessIdentityObservationFailed
  | AcnProcessIdentityObservationTimedOut

export const AcnCandidateLaunchFsm = FSM.defineFSM(
  {
    NotLaunched: AcnCandidateNotLaunched,
    Spawned: AcnCandidateSpawned,
    Admitted: AcnCandidateAdmitted,
    Ready: AcnCandidateReady,
    LaunchFailed: AcnCandidateLaunchFailed,
    ExitedBeforeAdmission: AcnCandidateExitedBeforeAdmission,
    ExitedAfterAdmission: AcnCandidateExitedAfterAdmission,
    AdmissionAcknowledgementLost: AcnCandidateAdmissionAcknowledgementLost,
    AdmissionExpired: AcnCandidateAdmissionExpired,
    LostAfterAdmission: AcnCandidateLostAfterAdmission,
  },
  {
    NotLaunched: ["Spawned", "LaunchFailed"],
    Spawned: ["Admitted", "ExitedBeforeAdmission", "AdmissionExpired", "AdmissionAcknowledgementLost"],
    Admitted: ["Ready", "ExitedAfterAdmission", "LostAfterAdmission"],
    Ready: [],
    LaunchFailed: [],
    ExitedBeforeAdmission: [],
    ExitedAfterAdmission: [],
    AdmissionAcknowledgementLost: [],
    AdmissionExpired: [],
    LostAfterAdmission: [],
  } as const,
)

interface CandidateRuntime {
  readonly process: ExactProcess
  readonly child: SpawnedAcnCandidate
}

export interface AcnCandidateLaunchSupervisor {
  readonly state: Effect.Effect<AcnCandidateLaunchState>
  readonly changes: Stream.Stream<AcnCandidateLaunchState>
  readonly launch: (
    command: Arr.NonEmptyReadonlyArray<string>,
  ) => Effect.Effect<void, AcnCandidateSupervisionError>
  readonly reconcile: (
    owner: Option.Option<AcnOwnerRecord>,
  ) => Effect.Effect<AcnCandidateLaunchState, AcnCandidateSupervisionError>
  readonly markReady: (instance: AcnInstance<AcnReady>) => Effect.Effect<void, AcnCandidateSupervisionError>
}

export const AcnCandidateLaunchSupervisor = Context.GenericTag<AcnCandidateLaunchSupervisor>(
  "@magnitudedev/sdk/AcnCandidateLaunchSupervisor",
)

const ownerNamesProcess = (owner: AcnOwnerRecord, process: ExactProcess): boolean =>
  owner.pid === process.pid && owner.processStartIdentity === process.processStartIdentity

const stopThenFail = <E>(
  child: SpawnedAcnCandidate,
  failure: E,
): Effect.Effect<never, E | AcnCandidateCleanupError> => child.stopAndReap.pipe(
  Effect.zipRight(Effect.fail(failure)),
)

export const makeAcnCandidateLaunchSupervisor = (
  spawner: ChildProcessSpawner,
  processes: ProcessGroupController,
): Effect.Effect<AcnCandidateLaunchSupervisor, never, Scope.Scope> => Effect.gen(function* () {
  const scope = yield* Scope.Scope
  const lifecycle = yield* SubscriptionRef.make<AcnCandidateLaunchState>(new AcnCandidateNotLaunched({}))
  const runtime = yield* Ref.make<Option.Option<CandidateRuntime>>(Option.none())
  const lock = yield* Effect.makeSemaphore(1)

  const launch: AcnCandidateLaunchSupervisor["launch"] = (command) => lock.withPermits(1)(
    Effect.gen(function* () {
      const now = yield* monotonicMillis
      const current = yield* SubscriptionRef.get(lifecycle)
      if (current._tag !== "NotLaunched") {
        return yield* new AcnCandidateLaunchAlreadyAttempted({})
      }
      const spawned = yield* spawner.spawn(command).pipe(
        Effect.provideService(Scope.Scope, scope),
        Effect.provideService(ProcessGroupController, processes),
        Effect.either,
      )
      if (spawned._tag === "Left") {
        yield* SubscriptionRef.set(lifecycle, AcnCandidateLaunchFsm.transition(current, "LaunchFailed", {
          failure: spawned.left,
        }))
        return yield* spawned.left
      }
      const child = spawned.right
      const inspected = yield* inspectExactProcess(processes, child.pid).pipe(Effect.either)
      if (inspected._tag === "Left") {
        yield* SubscriptionRef.set(lifecycle, AcnCandidateLaunchFsm.transition(current, "LaunchFailed", {
          failure: inspected.left,
        }))
        return yield* stopThenFail(child, inspected.left)
      }
      const identity = inspected.right
      if (Option.isNone(identity)) {
        const exit = yield* child.exited.pipe(Effect.timeoutOption(Duration.millis(100)))
        const failure = Option.match(exit, {
          onNone: () => new AcnCandidateIdentityUnavailable({ pid: child.pid }),
          onSome: ({ code, stderr }) => new AcnCandidateExitedBeforeIdentityObserved({
            pid: child.pid,
            code,
            stderr,
          }),
        })
        yield* SubscriptionRef.set(lifecycle, AcnCandidateLaunchFsm.transition(current, "LaunchFailed", { failure }))
        return yield* stopThenFail(child, failure)
      }
      const process = { pid: child.pid, processStartIdentity: identity.value }
      const confirmed = yield* child.confirmExactProcess(process).pipe(Effect.either)
      if (confirmed._tag === "Left") {
        yield* SubscriptionRef.set(lifecycle, AcnCandidateLaunchFsm.transition(current, "LaunchFailed", {
          failure: confirmed.left,
        }))
        return yield* stopThenFail(child, confirmed.left)
      }
      yield* Ref.set(runtime, Option.some({ process, child }))
      yield* SubscriptionRef.set(lifecycle, AcnCandidateLaunchFsm.transition(current, "Spawned", {
        process,
        launchedAt: now,
      }))
    }),
  )

  const reconcile: AcnCandidateLaunchSupervisor["reconcile"] = (owner) => lock.withPermits(1)(
    Effect.gen(function* () {
      const now = yield* monotonicMillis
      let current = yield* SubscriptionRef.get(lifecycle)
      const active = yield* Ref.get(runtime)
      if (Option.isNone(active)) return current
      const { child, process } = active.value

      if (current._tag === "Spawned" && Option.exists(owner, (value) => ownerNamesProcess(value, process))) {
        const admission = yield* child.admit.pipe(Effect.timeoutFail({
          duration: CANDIDATE_PARENT_RELEASE_TIMEOUT,
          onTimeout: () => new AcnCandidateAdmissionAcknowledgementTimedOut({ pid: process.pid }),
        }), Effect.either)
        if (admission._tag === "Left") {
          yield* SubscriptionRef.set(lifecycle, AcnCandidateLaunchFsm.transition(current, "AdmissionAcknowledgementLost", {
            lostAt: now,
          }))
          return yield* stopThenFail(child, admission.left)
        }
        current = AcnCandidateLaunchFsm.transition(current, "Admitted", { admittedAt: now })
        yield* SubscriptionRef.set(lifecycle, current)
      }

      if (current._tag === "Spawned") {
        const exited = yield* child.exited.pipe(Effect.timeoutOption(Duration.millis(1)))
        if (Option.isSome(exited)) {
          const next = AcnCandidateLaunchFsm.transition(current, "ExitedBeforeAdmission", {
            ...exited.value,
          })
          yield* SubscriptionRef.set(lifecycle, next)
          yield* child.stopAndReap
          return next
        }
        if (now - current.launchedAt >= Duration.toMillis(CANDIDATE_ADMISSION_TIMEOUT)) {
          const next = AcnCandidateLaunchFsm.transition(current, "AdmissionExpired", {})
          yield* SubscriptionRef.set(lifecycle, next)
          yield* child.stopAndReap
          return next
        }
      }

      if (current._tag === "Admitted" && !Option.exists(owner, (value) => ownerNamesProcess(value, process))) {
        const exited = yield* child.exited.pipe(Effect.timeoutOption(CANDIDATE_EXIT_DIAGNOSTIC_TIMEOUT))
        const next = Option.match(exited, {
          onNone: () => AcnCandidateLaunchFsm.transition(current, "LostAfterAdmission", {
            lostAt: now,
          }),
          onSome: (exit) => AcnCandidateLaunchFsm.transition(current, "ExitedAfterAdmission", {
            ...exit,
          }),
        })
        yield* SubscriptionRef.set(lifecycle, next)
        return next
      }

      if (current._tag === "Admitted") {
        const exited = yield* child.exited.pipe(Effect.timeoutOption(Duration.millis(1)))
        if (Option.isSome(exited)) {
          const next = AcnCandidateLaunchFsm.transition(current, "ExitedAfterAdmission", {
            ...exited.value,
          })
          yield* SubscriptionRef.set(lifecycle, next)
          return next
        }
      }
      return current
    }),
  )

  const markReady: AcnCandidateLaunchSupervisor["markReady"] = (instance) => lock.withPermits(1)(Effect.gen(function* () {
    const current = yield* SubscriptionRef.get(lifecycle)
    if (current._tag !== "Admitted") {
      return yield* new AcnCandidateReadyBeforeAdmission({})
    }
    if (current.process.pid !== instance.pid ||
      current.process.processStartIdentity !== instance.processStartIdentity) {
      return yield* new AcnCandidateReadyInstanceMismatch({
        candidate: current.process,
        ready: {
          pid: instance.pid,
          processStartIdentity: instance.processStartIdentity,
        },
      })
    }
    yield* SubscriptionRef.set(lifecycle, AcnCandidateLaunchFsm.transition(current, "Ready", {}))
  }))

  return AcnCandidateLaunchSupervisor.of({
    state: SubscriptionRef.get(lifecycle),
    changes: lifecycle.changes,
    launch,
    reconcile,
    markReady,
  })
})
const monotonicMillis = Clock.currentTimeNanos.pipe(
  Effect.map((nanos) => Number(nanos / 1_000_000n)),
)
