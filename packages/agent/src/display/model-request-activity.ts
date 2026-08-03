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
  phase: Schema.Literal('queued', 'preparing', 'prefill'),
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
  const { turn, progress } = observation
  const key = forkIdToKey(turn.forkId)

  if (progress.phase === 'generating') {
    const active = current.requests.get(key)
    if (!active) return current
    if (
      progress.requestId !== null
      && active.requestId !== null
      && active.requestId !== progress.requestId
    ) {
      return current
    }
    const requests = new Map(current.requests)
    requests.delete(key)
    return { requests }
  }

  if (progress.phase === 'cleared') {
    const active = current.requests.get(key)
    if (!active) return current
    if (
      progress.requestId !== null
      && active.requestId !== null
      && active.requestId !== progress.requestId
    ) {
      return current
    }
    const requests = new Map(current.requests)
    requests.delete(key)
    return { ...current, requests }
  }

  const nextActivity: ActiveModelRequest = {
    requestId: progress.requestId,
    turnId: turn.turnId,
    chainId: turn.chainId,
    forkId: turn.forkId,
    phase: progress.phase,
    completedTokens:
      progress.phase === 'prefill' ? progress.completedTokens : null,
    totalTokens: progress.phase === 'prefill' ? progress.totalTokens : null,
    cachedTokens: progress.phase === 'prefill' ? progress.cachedTokens : null,
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
