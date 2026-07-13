import { Context, Effect, Schema } from 'effect'
import { defineHarnessTool } from '@magnitudedev/harness'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import type { GoalState } from '../projections/goal'
import { ToolErrorSchema } from './errors'

const { ForkContext } = Fork

export interface GoalStateReader {
  readonly getState: () => Effect.Effect<GoalState>
}

export class GoalStateReaderTag extends Context.Tag('GoalStateReader')<
  GoalStateReaderTag,
  GoalStateReader
>() {}

export const GoalToolErrorSchema = ToolErrorSchema('GoalToolError', {})

const goalFail = (message: string) =>
  Effect.fail({ _tag: 'GoalToolError' as const, message })

export const finishGoalTool = defineHarnessTool({
  definition: {
    name: 'finish_goal',
    description: 'Mark the active goal complete. Use this only after verifying the full active goal against current evidence. Do not use it for partial progress, summaries, or plausible completion; provide concise evidence that proves the goal is complete.',
    inputSchema: Schema.Struct({
      evidence: Schema.String.annotations({
        description: 'Concise evidence that the full active goal is complete, such as verified requirements, tests run, files changed, worker results reviewed, or final answer delivered.',
      }),
    }),
    outputSchema: Schema.Struct({
      goalId: Schema.String,
      evidence: Schema.String,
    }),
  },
  errorSchema: GoalToolErrorSchema,
  execute: (input) =>
    Effect.gen(function* () {
      const { forkId } = yield* ForkContext
      if (forkId !== null) {
        return yield* goalFail('Only the root agent can finish the active goal.')
      }

      const evidence = input.evidence.trim()
      if (evidence.length === 0) {
        return yield* goalFail('Goal completion evidence is required.')
      }

      const goalReader = yield* GoalStateReaderTag
      const goalState = yield* goalReader.getState()
      if (!goalState.active) {
        return yield* goalFail('No active goal is currently running.')
      }

      const bus = yield* WorkerBusTag<AppEvent>()
      yield* bus.publish({
        type: 'goal_finished',
        forkId: null,
        goalId: goalState.active.goalId,
        evidence,
      })

      return {
        goalId: goalState.active.goalId,
        evidence,
      }
    }),
})
