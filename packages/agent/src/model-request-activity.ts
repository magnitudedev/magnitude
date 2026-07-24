import { Context, Effect, Layer, Stream, SubscriptionRef } from 'effect'
import { forkIdToKey } from '@magnitudedev/protocol'
import type { ModelRequestProgress } from '@magnitudedev/ai'

export interface ActiveModelRequest {
  readonly requestId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly forkId: string | null
  readonly startedAt: number
  readonly phase: 'queued' | 'preparing' | 'prefill'
  readonly completedTokens: number | null
  readonly totalTokens: number | null
  readonly cachedTokens: number | null
}

export type ActiveModelRequests = ReadonlyMap<string, ActiveModelRequest>

export interface ModelResponseTiming {
  readonly turnId: string
  readonly chainId: string
  readonly forkId: string | null
  readonly respondingSince: number
}

export type ModelResponseTimings = ReadonlyMap<string, ModelResponseTiming>

interface ModelRequestActivityState {
  readonly requests: ActiveModelRequests
  readonly responseTimings: ModelResponseTimings
}

export interface ModelRequestActivityService {
  readonly update: (
    turn: {
      readonly turnId: string
      readonly chainId: string
      readonly forkId: string | null
    },
    progress: ModelRequestProgress,
  ) => Effect.Effect<void>
  readonly get: Effect.Effect<ActiveModelRequests>
  readonly getResponseTimings: Effect.Effect<ModelResponseTimings>
  readonly changes: Stream.Stream<ModelRequestActivityState>
}

export class ModelRequestActivity extends Context.Tag('ModelRequestActivity')<
  ModelRequestActivity,
  ModelRequestActivityService
>() {}

export const ModelRequestActivityLive = Layer.effect(
  ModelRequestActivity,
  Effect.gen(function* () {
    const state = yield* SubscriptionRef.make<ModelRequestActivityState>({
      requests: new Map(),
      responseTimings: new Map(),
    })

    const update: ModelRequestActivityService['update'] = (turn, progress) =>
      SubscriptionRef.update(state, (current) => {
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
          const responseTimings = new Map(current.responseTimings)
          const timing = responseTimings.get(key)
          if (timing?.chainId !== turn.chainId) {
            responseTimings.set(key, {
              turnId: turn.turnId,
              chainId: turn.chainId,
              forkId: turn.forkId,
              respondingSince: Date.now(),
            })
          }
          return { requests, responseTimings }
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

        const active = current.requests.get(key)
        const requestId = progress.requestId
        const startedAt =
          active?.turnId === turn.turnId ? active.startedAt : Date.now()
        const nextActivity: ActiveModelRequest = {
          requestId,
          turnId: turn.turnId,
          chainId: turn.chainId,
          forkId: turn.forkId,
          startedAt,
          phase: progress.phase,
          completedTokens:
            progress.phase === 'prefill' ? progress.completedTokens : null,
          totalTokens: progress.phase === 'prefill' ? progress.totalTokens : null,
          cachedTokens: progress.phase === 'prefill' ? progress.cachedTokens : null,
        }
        const requests = new Map(current.requests)
        requests.set(key, nextActivity)
        const responseTiming = current.responseTimings.get(key)
        if (
          active?.turnId === turn.turnId
          || responseTiming?.chainId === turn.chainId
        ) {
          return { ...current, requests }
        }
        const responseTimings = new Map(current.responseTimings)
        responseTimings.delete(key)
        return { requests, responseTimings }
      })

    return ModelRequestActivity.of({
      update,
      get: SubscriptionRef.get(state).pipe(Effect.map((current) => current.requests)),
      getResponseTimings: SubscriptionRef.get(state).pipe(
        Effect.map((current) => current.responseTimings),
      ),
      changes: state.changes,
    })
  }),
)
