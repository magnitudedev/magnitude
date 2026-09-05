import {
  ProcessGroupController,
  type ExactProcess,
  type ProcessGroupStopError,
} from "@magnitudedev/acn-protocol/coordination"
import { Array as Arr, Context, Effect, Ref, Scope } from "effect"
import {
  type AcnCandidateBootstrapProcessExitUnproven,
  type AcnCandidateBootstrapProcessStopFailed,
  type AcnCandidateParentChannelReleaseFailed,
  type AcnCandidateSpawnFailed,
} from "./errors"

export interface AcnCandidateExit {
  readonly code: number
  readonly stderr: string
}

export type AcnCandidateCleanupError =
  | AcnCandidateBootstrapProcessStopFailed
  | AcnCandidateBootstrapProcessExitUnproven
  | ProcessGroupStopError

/** An ACN candidate whose cleanup remains armed until owner admission is observed. */
export interface SpawnedAcnCandidate {
  readonly pid: number
  readonly exited: Effect.Effect<AcnCandidateExit>
  readonly confirmExactProcess: (process: ExactProcess) => Effect.Effect<void>
  readonly admit: Effect.Effect<void, AcnCandidateParentChannelReleaseFailed>
  readonly stopAndReap: Effect.Effect<void, AcnCandidateCleanupError>
  /** Retires the exact admitted group when ownership or startup fails before readiness. */
  readonly retireAdmittedGroup: Effect.Effect<void, AcnCandidateCleanupError>
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
  | { readonly _tag: "Admitted"; readonly process: ExactProcess }
  | { readonly _tag: "Retired" }

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
          // A leader change proves the exact child's group is gone: cleanup's goal is met.
          yield* processes.stop({ leader: current.process })
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
          return yield* Effect.dieMessage(
            `observed ACN process ${process.pid} does not match spawned candidate PID ${candidate.pid}`,
          )
        }
        const current = yield* Ref.get(state)
        if (current._tag !== "AwaitingExactProcess") {
          return yield* Effect.dieMessage(
            `ACN candidate ${candidate.pid} exact process confirmed in state ${current._tag}`,
          )
        }
        yield* Ref.set(state, { _tag: "Armed", process })
      }))

    const admit = Effect.uninterruptibleMask((restore) => lock.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(state)
      if (current._tag !== "Armed") {
        return yield* Effect.dieMessage(
          `ACN candidate ${candidate.pid} admission attempted in state ${current._tag}`,
        )
      }
      yield* Ref.set(state, { _tag: "AdmissionAttempted", process: current.process })
      yield* restore(candidate.releaseParentChannel)
      yield* Ref.set(state, { _tag: "Admitted", process: current.process })
    })))

    const retireAdmittedGroup = lock.withPermits(1)(Effect.gen(function* () {
      const current = yield* Ref.get(state)
      if (current._tag === "Retired") return
      if (current._tag !== "Admitted") {
        return yield* Effect.dieMessage(
          `admitted ACN group retirement attempted in state ${current._tag}`,
        )
      }
      yield* processes.stop({ leader: current.process })
      yield* Ref.set(state, { _tag: "Retired" })
    }))

    return {
      pid: candidate.pid,
      exited: candidate.exited,
      confirmExactProcess,
      admit,
      stopAndReap,
      retireAdmittedGroup,
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
  "@magnitudedev/daemon-management/ChildProcessSpawner",
)
