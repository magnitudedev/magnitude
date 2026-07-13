import { Projection } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { AppEvent } from '../events'

export const ActiveGoalSchema = Schema.Struct({
  goalId: Schema.String,
  objective: Schema.String,
  startedAt: Schema.Number,
})
export type ActiveGoal = typeof ActiveGoalSchema.Type

export const FinishedGoalSchema = Schema.Struct({
  goalId: Schema.String,
  objective: Schema.String,
  startedAt: Schema.Number,
  finishedAt: Schema.Number,
  evidence: Schema.String,
})
export type FinishedGoal = typeof FinishedGoalSchema.Type

export const GoalStateSchema = Schema.Struct({
  active: Schema.NullOr(ActiveGoalSchema),
  finished: Schema.Array(FinishedGoalSchema),
})
export type GoalState = typeof GoalStateSchema.Type

export const GoalProjection = Projection.define<AppEvent>()({
  name: 'Goal',
  state: GoalStateSchema,

  initial: {
    active: null,
    finished: [],
  },

  eventHandlers: {
    goal_started: ({ event, state }) => ({
      ...state,
      active: {
        goalId: event.goalId,
        objective: event.objective,
        startedAt: event.timestamp,
      },
    }),

    goal_finished: ({ event, state }) => {
      const active = state.active
      if (!active || active.goalId !== event.goalId) return state

      return {
        active: null,
        finished: [
          ...state.finished,
          {
            ...active,
            finishedAt: event.timestamp,
            evidence: event.evidence,
          },
        ],
      }
    },
  },
})
