import { Clock, Duration, Effect } from "effect"
import type { ProcessGroupObservationFailed } from "./errors"
import type { ProcessGroupController, ProcessGroupObservation } from "./exact-process"
import type { ProcessGroup } from "./schemas"

/** Shared timing for process-group retirement. */
export const PROCESS_GROUP_EXIT_POLL_INTERVAL = Duration.millis(50)
export const PROCESS_GROUP_TERM_WAIT = Duration.seconds(2)
export const PROCESS_GROUP_KILL_WAIT = Duration.seconds(2)

/**
 * Polls until the process group is absent or the deadline elapses. The final
 * observation after the deadline is returned as the definitive result.
 */
export const waitForProcessGroupAbsence = (
  processes: ProcessGroupController,
  group: ProcessGroup,
  timeout: Duration.DurationInput,
): Effect.Effect<ProcessGroupObservation, ProcessGroupObservationFailed> =>
  Effect.gen(function* () {
    const deadline = (yield* Clock.currentTimeMillis) + Duration.toMillis(Duration.decode(timeout))
    while ((yield* Clock.currentTimeMillis) < deadline) {
      const observation = yield* processes.observeGroup(group)
      if (observation._tag === "ProcessGroupAbsent") return observation
      yield* Effect.sleep(PROCESS_GROUP_EXIT_POLL_INTERVAL)
    }
    return yield* processes.observeGroup(group)
  })
