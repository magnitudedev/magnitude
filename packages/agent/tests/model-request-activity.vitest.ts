import { describe, expect, it } from 'vitest'
import { Effect } from 'effect'
import {
  ModelRequestActivity,
  ModelRequestActivityLive,
} from '../src/model-request-activity'

describe('model request activity', () => {
  it('preserves its start time when admission hands off to the native request', async () => {
    const active = await Effect.runPromise(Effect.gen(function* () {
      const activity = yield* ModelRequestActivity
      const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }

      yield* activity.update(turn, { phase: 'preparing', requestId: null })
      const preparing = (yield* activity.get).get('root')
      yield* activity.update(turn, { phase: 'queued', requestId: 'request-1' })
      const queued = (yield* activity.get).get('root')
      return { preparing, queued }
    }).pipe(Effect.provide(ModelRequestActivityLive)))

    expect(active.preparing?.requestId).toBeNull()
    expect(active.queued).toMatchObject({
      requestId: 'request-1',
      phase: 'queued',
      startedAt: active.preparing?.startedAt,
    })
  })

  it('keeps newer activity when an older request clears late', async () => {
    const active = await Effect.runPromise(Effect.gen(function* () {
      const activity = yield* ModelRequestActivity
      const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }

      yield* activity.update(turn, { phase: 'queued', requestId: 'request-1' })
      yield* activity.update(turn, { phase: 'queued', requestId: 'request-2' })
      yield* activity.update(turn, { phase: 'cleared', requestId: 'request-1' })
      return yield* activity.get
    }).pipe(Effect.provide(ModelRequestActivityLive)))

    expect(active.get('root')).toMatchObject({ requestId: 'request-2' })
  })

  it('removes activity when generation starts', async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const activity = yield* ModelRequestActivity
      const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }

      yield* activity.update(turn, {
        phase: 'prefill',
        requestId: 'request-1',
        completedTokens: 9_400,
        totalTokens: 14_300,
        cachedTokens: 0,
      })
      yield* activity.update(turn, {
        phase: 'generating',
        requestId: 'request-1',
      })
      return {
        active: yield* activity.get,
        responseTimings: yield* activity.getResponseTimings,
      }
    }).pipe(Effect.provide(ModelRequestActivityLive)))

    expect(result.active.has('root')).toBe(false)
    expect(result.responseTimings.get('root')).toMatchObject({
      turnId: 'turn-1',
      forkId: null,
    })
  })

  it('retains the first response time across later requests in the same turn', async () => {
    const responseTimings = await Effect.runPromise(Effect.gen(function* () {
      const activity = yield* ModelRequestActivity
      const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }

      yield* activity.update(turn, { phase: 'preparing', requestId: null })
      yield* activity.update(turn, { phase: 'generating', requestId: 'request-1' })
      const first = (yield* activity.getResponseTimings).get('root')
      const nextTurn = { ...turn, turnId: 'turn-2' }
      yield* activity.update(nextTurn, { phase: 'preparing', requestId: null })
      yield* activity.update(nextTurn, { phase: 'generating', requestId: 'request-2' })
      const second = (yield* activity.getResponseTimings).get('root')
      return { first, second }
    }).pipe(Effect.provide(ModelRequestActivityLive)))

    expect(responseTimings.first?.respondingSince).toBeTypeOf('number')
    expect(responseTimings.second?.respondingSince).toBe(
      responseTimings.first?.respondingSince,
    )
  })
})
