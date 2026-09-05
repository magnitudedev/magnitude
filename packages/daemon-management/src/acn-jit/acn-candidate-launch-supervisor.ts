import { FSM } from "@magnitudedev/utils"
import { AcnReady, type AcnInstance } from "@magnitudedev/acn-protocol"
import {
  ExactProcessSchema,
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
import { type AcnCandidateCleanupError, type ChildProcessSpawner, type SpawnedAcnCandidate } from "./child-process"
import {
  AcnCandidateAdmissionTimedOut,
  AcnCandidateExitedAfterAdmission,
  AcnCandidateExitedBeforeAdmission,
  AcnCandidateFailureSchema,
  AcnCandidateIdentityUnavailable,
  AcnCandidateOwnershipLost,
  AcnCandidateParentChannelReleaseFailed,
  type AcnCandidateFailure,
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
export class AcnCandidateFailed extends Schema.TaggedClass<AcnCandidateFailed>()(
  "Failed", { failure: AcnCandidateFailureSchema },
) {}

export const AcnCandidateLaunchStateSchema = Schema.Union(
  AcnCandidateNotLaunched,
  AcnCandidateSpawned,
  AcnCandidateAdmitted,
  AcnCandidateReady,
  AcnCandidateFailed,
)
export type AcnCandidateLaunchState = typeof AcnCandidateLaunchStateSchema.Type

export const AcnCandidateLaunchFsm = FSM.defineFSM(
  {
    NotLaunched: AcnCandidateNotLaunched,
    Spawned: AcnCandidateSpawned,
    Admitted: AcnCandidateAdmitted,
    Ready: AcnCandidateReady,
    Failed: AcnCandidateFailed,
  },
  {
    NotLaunched: ["Spawned", "Failed"],
    Spawned: ["Admitted", "Failed"],
    Admitted: ["Ready", "Failed"],
    Ready: [],
    Failed: [],
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
  ) => Effect.Effect<void, AcnCandidateCleanupError>
  readonly reconcile: (
    owner: Option.Option<AcnOwnerRecord>,
  ) => Effect.Effect<AcnCandidateLaunchState, AcnCandidateCleanupError>
  readonly markReady: (instance: AcnInstance<AcnReady>) => Effect.Effect<void>
}

export const AcnCandidateLaunchSupervisor = Context.GenericTag<AcnCandidateLaunchSupervisor>(
  "@magnitudedev/daemon-management/AcnCandidateLaunchSupervisor",
)

const ownerNamesProcess = (owner: AcnOwnerRecord, process: ExactProcess): boolean =>
  owner.pid === process.pid && owner.processStartIdentity === process.processStartIdentity

const monotonicMillis = Clock.currentTimeNanos.pipe(
  Effect.map((nanos) => Number(nanos / 1_000_000n)),
)

export const makeAcnCandidateLaunchSupervisor = (
  spawner: ChildProcessSpawner,
  processes: ProcessGroupController,
): Effect.Effect<AcnCandidateLaunchSupervisor, never, Scope.Scope> => Effect.gen(function* () {
  const scope = yield* Scope.Scope
  const lifecycle = yield* SubscriptionRef.make<AcnCandidateLaunchState>(new AcnCandidateNotLaunched({}))
  const runtime = yield* Ref.make<Option.Option<CandidateRuntime>>(Option.none())
  const lock = yield* Effect.makeSemaphore(1)

  const failFrom = (
    from: AcnCandidateLaunchState,
    failure: AcnCandidateFailure,
  ): Effect.Effect<AcnCandidateFailed> => Effect.gen(function* () {
    if (from._tag === "Ready" || from._tag === "Failed") {
      return yield* Effect.dieMessage(`ACN candidate failure recorded in terminal state ${from._tag}`)
    }
    const next = AcnCandidateLaunchFsm.transition(from, "Failed", { failure })
    yield* SubscriptionRef.set(lifecycle, next)
    return next
  })

  const launch: AcnCandidateLaunchSupervisor["launch"] = (command) => lock.withPermits(1)(
    Effect.gen(function* () {
      const now = yield* monotonicMillis
      const current = yield* SubscriptionRef.get(lifecycle)
      if (current._tag !== "NotLaunched") {
        return yield* Effect.dieMessage(`ACN candidate launch attempted in state ${current._tag}`)
      }
      const spawned = yield* spawner.spawn(command).pipe(
        Effect.provideService(Scope.Scope, scope),
        Effect.provideService(ProcessGroupController, processes),
        Effect.either,
      )
      if (spawned._tag === "Left") {
        yield* failFrom(current, spawned.left)
        return
      }
      const child = spawned.right
      const inspected = yield* inspectExactProcess(processes, child.pid).pipe(Effect.either)
      if (inspected._tag === "Left") {
        yield* failFrom(current, inspected.left)
        return yield* child.stopAndReap
      }
      const identity = inspected.right
      if (Option.isNone(identity)) {
        const exit = yield* child.exited.pipe(Effect.timeoutOption(Duration.millis(100)))
        yield* failFrom(current, Option.match(exit, {
          onNone: () => new AcnCandidateIdentityUnavailable({ pid: child.pid }),
          onSome: ({ code, stderr }) => new AcnCandidateExitedBeforeAdmission({
            pid: child.pid,
            code,
            stderr,
          }),
        }))
        return yield* child.stopAndReap
      }
      const process = identity.value
      yield* child.confirmExactProcess(process)
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
      if (Option.isNone(active) || current._tag === "Failed" || current._tag === "Ready") return current
      const { child, process } = active.value

      if (current._tag === "Spawned" && Option.exists(owner, (value) => ownerNamesProcess(value, process))) {
        const admission = yield* child.admit.pipe(Effect.timeoutFail({
          duration: CANDIDATE_PARENT_RELEASE_TIMEOUT,
          onTimeout: () => new AcnCandidateParentChannelReleaseFailed({
            pid: process.pid,
            message: "parent channel release timed out",
          }),
        }), Effect.either)
        if (admission._tag === "Left") {
          const failed = yield* failFrom(current, admission.left)
          yield* child.stopAndReap
          return failed
        }
        current = AcnCandidateLaunchFsm.transition(current, "Admitted", { admittedAt: now })
        yield* SubscriptionRef.set(lifecycle, current)
      }

      if (current._tag === "Spawned") {
        const exited = yield* child.exited.pipe(Effect.timeoutOption(Duration.millis(1)))
        if (Option.isSome(exited)) {
          const failed = yield* failFrom(current, new AcnCandidateExitedBeforeAdmission({
            pid: process.pid,
            ...exited.value,
          }))
          yield* child.stopAndReap
          return failed
        }
        if (now - current.launchedAt >= Duration.toMillis(CANDIDATE_ADMISSION_TIMEOUT)) {
          const failed = yield* failFrom(current, new AcnCandidateAdmissionTimedOut({ pid: process.pid }))
          yield* child.stopAndReap
          return failed
        }
      }

      if (current._tag === "Admitted" && !Option.exists(owner, (value) => ownerNamesProcess(value, process))) {
        const exited = yield* child.exited.pipe(Effect.timeoutOption(CANDIDATE_EXIT_DIAGNOSTIC_TIMEOUT))
        yield* child.retireAdmittedGroup
        const failed = yield* failFrom(current, Option.match(exited, {
          onNone: () => new AcnCandidateOwnershipLost({ pid: process.pid }),
          onSome: (exit): AcnCandidateFailure => new AcnCandidateExitedAfterAdmission({
            pid: process.pid,
            ...exit,
          }),
        }))
        return failed
      }

      if (current._tag === "Admitted") {
        const exited = yield* child.exited.pipe(Effect.timeoutOption(Duration.millis(1)))
        if (Option.isSome(exited)) {
          yield* child.retireAdmittedGroup
          const failed = yield* failFrom(current, new AcnCandidateExitedAfterAdmission({
            pid: process.pid,
            ...exited.value,
          }))
          return failed
        }
      }
      return current
    }),
  )

  const markReady: AcnCandidateLaunchSupervisor["markReady"] = (instance) => lock.withPermits(1)(Effect.gen(function* () {
    const current = yield* SubscriptionRef.get(lifecycle)
    if (current._tag !== "Admitted") {
      return yield* Effect.dieMessage(`ACN candidate marked ready in state ${current._tag}`)
    }
    if (current.process.pid !== instance.pid ||
      current.process.processStartIdentity !== instance.processStartIdentity) {
      return yield* Effect.dieMessage(
        `ready ACN ${instance.pid} is not the supervised candidate ${current.process.pid}`,
      )
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
