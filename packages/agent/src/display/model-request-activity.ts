import { Projection } from '@magnitudedev/event-core'
import { forkIdToKey } from '@magnitudedev/acn-protocol'
import { Schema } from 'effect'
import type { AppEvent } from '../events'
import {
  ModelRequestActivityAmbient,
  type ModelRequestActivityObservation,
} from '../model/model-request-activity'

export {
  ModelRequestActivityAmbient,
  type ModelRequestActivityObservation,
} from '../model/model-request-activity'

const ActiveModelRequestSchema = Schema.Struct({
  requestId: Schema.NullOr(Schema.String),
  turnId: Schema.String,
  chainId: Schema.String,
  forkId: Schema.NullOr(Schema.String),
  phase: Schema.Literal('model_loading', 'queued', 'preparing', 'prefill'),
  completedTokens: Schema.NullOr(Schema.Number),
  totalTokens: Schema.NullOr(Schema.Number),
  cachedTokens: Schema.NullOr(Schema.Number),
})

export type ActiveModelRequest = typeof ActiveModelRequestSchema.Type

export type ActiveModelRequests = ReadonlyMap<string, ActiveModelRequest>

export const ModelRequestActivityStateSchema = Schema.Struct({
  requests: Schema.ReadonlyMap({
    key: Schema.String,
    value: ActiveModelRequestSchema,
  }),
})

export type ModelRequestActivityState =
  typeof ModelRequestActivityStateSchema.Type

export const initialModelRequestActivityState = (): ModelRequestActivityState => ({
  requests: new Map(),
})

export function reduceModelRequestActivity(
  current: ModelRequestActivityState,
  observation: ModelRequestActivityObservation,
): ModelRequestActivityState {
  const { turn, activity } = observation
  const key = forkIdToKey(turn.forkId)

  if (activity._tag === 'Ended') {
    const active = current.requests.get(key)
    if (!active) return current
    if (
      activity.requestId !== null
      && active.requestId !== null
      && active.requestId !== activity.requestId
    ) {
      return current
    }
    const requests = new Map(current.requests)
    requests.delete(key)
    return { requests }
  }

  if (activity._tag === 'Streaming') {
    const active = current.requests.get(key)
    if (!active) return current
    if (
      activity.requestId !== null
      && active.requestId !== null
      && active.requestId !== activity.requestId
    ) {
      return current
    }
    const requests = new Map(current.requests)
    requests.delete(key)
    return { requests }
  }

  if (activity._tag === 'Starting') return current

  const preparation = activity.preparation

  const nextActivity: ActiveModelRequest = {
    requestId: activity.requestId,
    turnId: turn.turnId,
    chainId: turn.chainId,
    forkId: turn.forkId,
    phase: preparation.phase,
    completedTokens:
      preparation.phase === 'prefill' ? preparation.completed_tokens : null,
    totalTokens: preparation.phase === 'prefill' ? preparation.total_tokens : null,
    cachedTokens: preparation.phase === 'prefill' ? preparation.cached_tokens : null,
  }
  const requests = new Map(current.requests)
  requests.set(key, nextActivity)
  return { requests }
}

export const ModelRequestActivityProjection = Projection.define<AppEvent>()({
  name: 'ModelRequestActivity',
  state: ModelRequestActivityStateSchema,
  initial: initialModelRequestActivityState(),
  ambients: [ModelRequestActivityAmbient] as const,
  ambientHandlers: (on) => [
    on(ModelRequestActivityAmbient, ({ value, state }) =>
      value === null ? state : reduceModelRequestActivity(state, value)),
  ],
})
