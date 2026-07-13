import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'

/**
 * Ambient for the initial task provided at session start.
 *
 * When set (non-null), the session was started with a specific task
 * (e.g. via --prompt). The autopilot uses the task-driving prompt
 * instead of the user-imitation prompt when a task is present.
 *
 * Not set (null) by default — the regular user-imitation autopilot is used.
 */
export const InitialTaskAmbient = Ambient.define<string | null, never>({
  name: 'InitialTask',
  initial: Effect.succeed(null),
})

export function publishInitialTask(task: string | null) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    yield* ambientService.update(InitialTaskAmbient, task)
  })
}
