import { Array as Arr, Context, Effect, Ref, Scope } from "effect"
import { AcnEnsuranceFailed } from "./errors"

/** An ACN candidate child whose lifetime is owned by the spawning scope until handoff. */
export interface SpawnedAcnCandidate {
  readonly pid: number

  /**
   * Transfers lifetime ownership to the exact candidate already committed in
   * ACN process state. This is one-shot and leaves scoped cleanup disarmed.
   */
  readonly handoff: Effect.Effect<void, AcnEnsuranceFailed>
}

interface PreHandoffAcnCandidate {
  readonly pid: number
  readonly releaseForHandoff: Effect.Effect<void, AcnEnsuranceFailed>
  readonly stopAndReap: Effect.Effect<void, AcnEnsuranceFailed>
}

/** Installs the ownership guarantee shared by every platform spawner. */
export const scopePreHandoffCandidate = (
  candidate: PreHandoffAcnCandidate,
): Effect.Effect<SpawnedAcnCandidate, never, Scope.Scope> =>
  Effect.gen(function* () {
    const handedOff = yield* Ref.make(false)
    const handoffAttempted = yield* Ref.make(false)
    yield* Effect.addFinalizer(() =>
      Ref.get(handedOff).pipe(
        Effect.flatMap((complete) => (complete ? Effect.void : candidate.stopAndReap)),
        Effect.orDie,
      ),
    )
    const handoff = Effect.uninterruptible(
      Effect.gen(function* () {
        const alreadyAttempted = yield* Ref.getAndSet(handoffAttempted, true)
        if (alreadyAttempted) {
          return yield* new AcnEnsuranceFailed({
            reason: `ACN candidate ${candidate.pid} handoff was already attempted`,
          })
        }
        yield* candidate.releaseForHandoff
        yield* Ref.set(handedOff, true)
      }),
    )
    return {
      pid: candidate.pid,
      handoff,
    }
  })

export interface ChildProcessSpawner {
  readonly spawn: (
    command: Arr.NonEmptyReadonlyArray<string>,
  ) => Effect.Effect<SpawnedAcnCandidate, AcnEnsuranceFailed, Scope.Scope>
}

export const ChildProcessSpawner = Context.GenericTag<ChildProcessSpawner>(
  "@magnitudedev/sdk/ChildProcessSpawner",
)
