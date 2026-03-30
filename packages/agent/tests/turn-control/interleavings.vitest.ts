import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { assertNoTurnIdMismatch, eventsForFork, mkTurnCompletedSuccess, mkTurnStarted } from './helpers'

describe('turn control interleavings', () => {
  it.live('delayed completion after a new start is detected as mismatch', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-old', chainId: 'c-race' }))
      yield* h.send(mkTurnStarted({ turnId: 't-new', chainId: 'c-race' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-old', chainId: 'c-race' }))

      expect(() => assertNoTurnIdMismatch(eventsForFork(h, null))).toThrow()
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('wake pressure around completion boundary keeps IDs consistent', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({ type: 'wake', forkId: null })
      yield* h.send(mkTurnStarted({ turnId: 't-rp-1', chainId: 'c-rp' }))
      yield* h.send({ type: 'wake', forkId: null })
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-rp-1', chainId: 'c-rp', result: { success: true, turnDecision: 'continue' } }))
      yield* h.send(mkTurnStarted({ turnId: 't-rp-2', chainId: 'c-rp' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-rp-2', chainId: 'c-rp' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('subagent completion and parent scheduling do not cross-contaminate IDs', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ forkId: null, turnId: 'root-1', chainId: 'root-c' }))
      yield* h.send(mkTurnStarted({ forkId: 'fork-b', turnId: 'sub-1', chainId: 'sub-c' }))
      yield* h.send(mkTurnCompletedSuccess({ forkId: 'fork-b', turnId: 'sub-1', chainId: 'sub-c' }))
      yield* h.send({ type: 'wake', forkId: null })
      yield* h.send(mkTurnCompletedSuccess({ forkId: null, turnId: 'root-1', chainId: 'root-c' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
      assertNoTurnIdMismatch(eventsForFork(h, 'fork-b'))
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('sanity: helper catches synthetic stale completion', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-good', chainId: 'c-neg' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-wrong', chainId: 'c-neg' }))

      expect(() => assertNoTurnIdMismatch(eventsForFork(h, null))).toThrow()
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
