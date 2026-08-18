import { Clock, Duration, Effect } from "effect"
import type { ExactProcessInspectionFailed } from "./errors"
import type { ExactProcessController } from "./exact-process"
import type { ExactProcess } from "./schemas"

/** Shared timing for exact owner-tree retirement. */
export const TREE_EXIT_POLL_INTERVAL = Duration.millis(50)
export const TREE_TERM_WAIT = Duration.seconds(2)
export const TREE_KILL_WAIT = Duration.seconds(2)

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
