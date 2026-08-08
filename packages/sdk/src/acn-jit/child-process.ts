import { Array as Arr, Context, Effect, Ref, Scope } from "effect"
import { AcnEnsuranceFailed } from "./errors"

export interface AcnCandidateExit {
  readonly code: number
  readonly stderr: string
}

/** An ACN candidate whose cleanup remains armed until owner admission is observed. */
export interface SpawnedAcnCandidate {
  readonly pid: number
  readonly exited: Effect.Effect<AcnCandidateExit>
  readonly admit: Effect.Effect<void, AcnEnsuranceFailed>
}

interface ScopedAcnCandidate {
  readonly pid: number
  readonly exited: Effect.Effect<AcnCandidateExit>
  readonly stopAndReap: Effect.Effect<void, AcnEnsuranceFailed>
  readonly releaseParentChannel: Effect.Effect<void, AcnEnsuranceFailed>
}

/** Installs the candidate cleanup/admission boundary shared by platform spawners. */
export const scopeAcnCandidate = (
  candidate: ScopedAcnCandidate,
): Effect.Effect<SpawnedAcnCandidate, never, Scope.Scope> =>
  Effect.gen(function* () {
    const state = yield* Ref.make<"Armed" | "AdmissionAttempted" | "Admitted">("Armed")
    yield* Effect.addFinalizer(() => Ref.get(state).pipe(
      Effect.flatMap((value) => value === "Admitted" ? Effect.void : candidate.stopAndReap),
      Effect.orDie,
    ))
    const admit = Effect.uninterruptibleMask((restore) => Effect.gen(function* () {
      const firstAttempt = yield* Ref.modify(state, (current) => current === "Armed"
        ? [true, "AdmissionAttempted" as const]
        : [false, current])
      if (!firstAttempt) {
        return yield* new AcnEnsuranceFailed({
          reason: `ACN candidate ${candidate.pid} admission was already acknowledged`,
        })
      }
      yield* restore(candidate.releaseParentChannel)
      yield* Ref.set(state, "Admitted")
    }))
    return { pid: candidate.pid, exited: candidate.exited, admit }
  })

export interface ChildProcessSpawner {
  readonly spawn: (
    command: Arr.NonEmptyReadonlyArray<string>,
  ) => Effect.Effect<SpawnedAcnCandidate, AcnEnsuranceFailed, Scope.Scope>
}

export const ChildProcessSpawner = Context.GenericTag<ChildProcessSpawner>(
  "@magnitudedev/sdk/ChildProcessSpawner",
)
