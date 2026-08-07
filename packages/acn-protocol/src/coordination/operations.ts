import { Clock, Duration, Effect } from "effect"
import { AcnProcessStoreUnavailable, type AcnProcessStoreError, type ExactProcessInspectionFailed } from "./errors"
import type { ExactProcessController } from "./exact-process"
import type { ExactProcess } from "./schemas"

/**
 * Shared coordination timing. Every consumer of the coordination layer uses
 * these same durations so the reap and retry policies are consistent by
 * construction.
 */
export const COORDINATION_POLL_INTERVAL = Duration.seconds(1)
export const TREE_EXIT_POLL_INTERVAL = Duration.millis(50)
export const TREE_TERM_WAIT = Duration.seconds(2)
export const TREE_KILL_WAIT = Duration.seconds(2)

/**
 * Retries a coordination-store observation whenever the store is temporarily
 * unavailable. `AcnProcessStoreUnavailable` is the only transient failure —
 * `AcnProcessStoreInvalid` is a permanent corruption signal that callers must
 * surface.
 */
export const retryStoreObservation = <A, R>(
  effect: Effect.Effect<A, AcnProcessStoreError, R>,
): Effect.Effect<A, AcnProcessStoreError, R> => Effect.suspend(() => effect.pipe(
  Effect.catchTag("AcnProcessStoreUnavailable", (error) =>
    Effect.logWarning("ACN coordination observation is temporarily unavailable").pipe(
      Effect.annotateLogs({ operation: error.operation, path: error.path }),
      Effect.zipRight(Effect.sleep(COORDINATION_POLL_INTERVAL)),
      Effect.zipRight(retryStoreObservation(effect)),
    )),
))

/**
 * Polls `treeAbsent` until it returns `true` or the deadline elapses. The final
 * probe after the deadline returns the definitive answer.
 */
export const waitForTreeAbsence = (
  processes: ExactProcessController,
  process: ExactProcess,
  timeout: Duration.DurationInput,
): Effect.Effect<boolean, ExactProcessInspectionFailed> =>
  Effect.gen(function* () {
    const deadline = (yield* Clock.currentTimeMillis) + Duration.toMillis(Duration.decode(timeout))
    while ((yield* Clock.currentTimeMillis) < deadline) {
      if (yield* processes.treeAbsent(process)) return true
      yield* Effect.sleep(TREE_EXIT_POLL_INTERVAL)
    }
    return yield* processes.treeAbsent(process)
  })

/**
 * Reaps one exact process tree with a graceful first attempt: signal `term`,
 * wait, then escalate to `kill` and wait again. Returns `true` when the tree
 * is proven absent, `false` when it could not be proven absent within the
 * bounded waits. Callers map the `false` outcome to their own typed failure.
 */
export const reapProcessTree = (
  processes: ExactProcessController,
  process: ExactProcess,
): Effect.Effect<boolean, ExactProcessInspectionFailed> =>
  Effect.gen(function* () {
    if (yield* processes.treeAbsent(process)) return true
    yield* processes.signalTree(process, "term")
    if (yield* waitForTreeAbsence(processes, process, TREE_TERM_WAIT)) return true
    yield* processes.signalTree(process, "kill")
    return yield* waitForTreeAbsence(processes, process, TREE_KILL_WAIT)
  })
