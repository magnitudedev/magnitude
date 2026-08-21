import {
  PROCESS_GROUP_KILL_WAIT,
  PROCESS_GROUP_TERM_WAIT,
  ProcessGroupController,
  waitForProcessGroupAbsence,
  type ExactProcess,
  type ExactProcessIdentityObservationFailed,
  type ProcessGroup,
  type ProcessGroupSignalOutcome,
} from "@magnitudedev/acn-protocol/coordination"
import { Array as Arr, Context, Effect, Ref, Scope } from "effect"
import {
  AcnCandidateAdmissionAlreadyAcknowledged,
  AcnCandidateAdmissionBeforeExactProcessConfirmed,
  type AcnCandidateBootstrapProcessExitUnproven,
  type AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateExactProcessAlreadyConfirmed,
  AcnCandidateExactProcessPidMismatch,
  type AcnCandidateParentChannelReleaseFailed,
  AcnCandidateProcessGroupAbsenceUnproven,
  AcnCandidateProcessGroupKillFailed,
  AcnCandidateProcessGroupKillPermissionDenied,
  AcnCandidateProcessGroupLeaderChanged,
  AcnCandidateProcessGroupObservationFailed,
  AcnCandidateProcessGroupTerminationFailed,
  AcnCandidateProcessGroupTerminationPermissionDenied,
  type AcnCandidateSpawnFailed,
} from "./errors"

export interface AcnCandidateExit {
  readonly code: number
  readonly stderr: string
}

export type AcnCandidateCleanupError =
  | AcnCandidateBootstrapProcessStopFailed
  | AcnCandidateBootstrapProcessExitUnproven
  | ExactProcessIdentityObservationFailed
  | AcnCandidateProcessGroupObservationFailed
  | AcnCandidateProcessGroupTerminationPermissionDenied
  | AcnCandidateProcessGroupTerminationFailed
  | AcnCandidateProcessGroupKillPermissionDenied
  | AcnCandidateProcessGroupKillFailed
  | AcnCandidateProcessGroupLeaderChanged
  | AcnCandidateProcessGroupAbsenceUnproven

export type AcnCandidateExactProcessConfirmationError =
  | AcnCandidateExactProcessPidMismatch
  | AcnCandidateExactProcessAlreadyConfirmed

/** An ACN candidate whose cleanup remains armed until owner admission is observed. */
export interface SpawnedAcnCandidate {
  readonly pid: number
  readonly exited: Effect.Effect<AcnCandidateExit>
  readonly confirmExactProcess: (
    process: ExactProcess,
  ) => Effect.Effect<void, AcnCandidateExactProcessConfirmationError>
  readonly admit: Effect.Effect<
    void,
    | AcnCandidateAdmissionBeforeExactProcessConfirmed
    | AcnCandidateAdmissionAlreadyAcknowledged
    | AcnCandidateParentChannelReleaseFailed
  >
  readonly stopAndReap: Effect.Effect<void, AcnCandidateCleanupError>
}

interface ScopedAcnCandidate {
  readonly pid: number
  readonly exited: Effect.Effect<AcnCandidateExit>
  /** Stops only the exact raw child handle, before process identity has been confirmed. */
  readonly stopBootstrapProcess: Effect.Effect<
    void,
    AcnCandidateBootstrapProcessStopFailed | AcnCandidateBootstrapProcessExitUnproven
  >
  readonly releaseParentChannel: Effect.Effect<void, AcnCandidateParentChannelReleaseFailed>
}

type CandidateOwnershipState =
  | { readonly _tag: "AwaitingExactProcess" }
  | { readonly _tag: "Armed"; readonly process: ExactProcess }
  | { readonly _tag: "AdmissionAttempted"; readonly process: ExactProcess }
  | { readonly _tag: "Admitted" }
  | { readonly _tag: "Retired" }

const groupFrom = (process: ExactProcess): ProcessGroup => ({ leader: process })

const candidateObservationFailure = (
  process: ExactProcess,
  message: string,
) => new AcnCandidateProcessGroupObservationFailed({ pid: process.pid, message })

const classifySignal = (
  process: ExactProcess,
  result: ProcessGroupSignalOutcome,
): Effect.Effect<void, AcnCandidateProcessGroupLeaderChanged> => {
  if (result._tag === "ProcessGroupLeaderChanged") {
    return Effect.fail(new AcnCandidateProcessGroupLeaderChanged({
      candidate: process,
      observedLeader: result.observedLeader,
    }))
  }
  return Effect.void
}

const stopExactProcessGroup = (
  processes: ProcessGroupController,
  process: ExactProcess,
): Effect.Effect<void, AcnCandidateCleanupError> => Effect.gen(function* () {
  const group = groupFrom(process)
  const initial = yield* processes.observeGroup(group).pipe(
    Effect.mapError((error) => candidateObservationFailure(process, error.message)),
  )
  if (initial._tag === "ProcessGroupAbsent") return

  const terminated = yield* processes.signalGroup(group, "term").pipe(
    Effect.mapError((error) => {
      switch (error._tag) {
        case "ExactProcessIdentityObservationFailed": return error
        case "ProcessGroupSignalPermissionDenied":
          return new AcnCandidateProcessGroupTerminationPermissionDenied({
            pid: process.pid,
            message: error.message,
          })
        case "ProcessGroupSignalFailed":
          return new AcnCandidateProcessGroupTerminationFailed({ pid: process.pid, message: error.message })
      }
    }),
  )
  yield* classifySignal(process, terminated)
  if (terminated._tag === "ProcessGroupAlreadyAbsent") return

  const afterTerm = yield* waitForProcessGroupAbsence(processes, group, PROCESS_GROUP_TERM_WAIT).pipe(
    Effect.mapError((error) => candidateObservationFailure(process, error.message)),
  )
  if (afterTerm._tag === "ProcessGroupAbsent") return

  const killed = yield* processes.signalGroup(group, "kill").pipe(
    Effect.mapError((error) => {
      switch (error._tag) {
        case "ExactProcessIdentityObservationFailed": return error
        case "ProcessGroupSignalPermissionDenied":
          return new AcnCandidateProcessGroupKillPermissionDenied({ pid: process.pid, message: error.message })
        case "ProcessGroupSignalFailed":
          return new AcnCandidateProcessGroupKillFailed({ pid: process.pid, message: error.message })
      }
    }),
  )
  yield* classifySignal(process, killed)
  if (killed._tag === "ProcessGroupAlreadyAbsent") return

  const afterKill = yield* waitForProcessGroupAbsence(processes, group, PROCESS_GROUP_KILL_WAIT).pipe(
    Effect.mapError((error) => candidateObservationFailure(process, error.message)),
  )
  if (afterKill._tag === "ProcessGroupPresent") {
    return yield* new AcnCandidateProcessGroupAbsenceUnproven({ pid: process.pid })
  }
})

/** Installs the candidate cleanup/admission boundary shared by platform spawners. */
export const scopeAcnCandidate = (
  candidate: ScopedAcnCandidate,
): Effect.Effect<SpawnedAcnCandidate, never, Scope.Scope | ProcessGroupController> =>
  Effect.gen(function* () {
    const processes = yield* ProcessGroupController
    const state = yield* Ref.make<CandidateOwnershipState>({ _tag: "AwaitingExactProcess" })
    const lock = yield* Effect.makeSemaphore(1)

    const stopAndReap = lock.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(state)
      switch (current._tag) {
        case "Admitted":
        case "Retired":
          return
        case "AwaitingExactProcess":
          yield* candidate.stopBootstrapProcess
          break
        case "Armed":
        case "AdmissionAttempted":
          yield* stopExactProcessGroup(processes, current.process)
          break
      }
      yield* Ref.set(state, { _tag: "Retired" })
    }))

    yield* Effect.addFinalizer(() => stopAndReap.pipe(
      Effect.catchAll((error) => Effect.logError("Failed to stop and reap unadmitted ACN candidate", error)),
    ))

    const confirmExactProcess: SpawnedAcnCandidate["confirmExactProcess"] = (process) =>
      lock.withPermits(1)(Effect.gen(function* () {
        if (process.pid !== candidate.pid) {
          return yield* new AcnCandidateExactProcessPidMismatch({
            candidatePid: candidate.pid,
            observed: process,
          })
        }
        const current = yield* Ref.get(state)
        if (current._tag === "AwaitingExactProcess") {
          yield* Ref.set(state, { _tag: "Armed", process })
          return
        }
        if (current._tag === "Armed" &&
          current.process.processStartIdentity === process.processStartIdentity) return
        const confirmed = current._tag === "Armed" || current._tag === "AdmissionAttempted"
          ? current.process
          : process
        return yield* new AcnCandidateExactProcessAlreadyConfirmed({ confirmed, attempted: process })
      }))

    const admit = Effect.uninterruptibleMask((restore) => lock.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(state)
      if (current._tag === "AwaitingExactProcess") {
        return yield* new AcnCandidateAdmissionBeforeExactProcessConfirmed({ pid: candidate.pid })
      }
      if (current._tag !== "Armed") {
        return yield* new AcnCandidateAdmissionAlreadyAcknowledged({ pid: candidate.pid })
      }
      yield* Ref.set(state, { _tag: "AdmissionAttempted", process: current.process })
      yield* restore(candidate.releaseParentChannel)
      yield* Ref.set(state, { _tag: "Admitted" })
    })))

    return {
      pid: candidate.pid,
      exited: candidate.exited,
      confirmExactProcess,
      admit,
      stopAndReap,
    }
  })

export interface ChildProcessSpawner {
  readonly spawn: (
    command: Arr.NonEmptyReadonlyArray<string>,
  ) => Effect.Effect<
    SpawnedAcnCandidate,
    AcnCandidateSpawnFailed,
    Scope.Scope | ProcessGroupController
  >
}

export const ChildProcessSpawner = Context.GenericTag<ChildProcessSpawner>(
  "@magnitudedev/sdk/ChildProcessSpawner",
)
