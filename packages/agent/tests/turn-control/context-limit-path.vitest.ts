import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { CompactionProjection } from '../../src/projections/compaction'
import {
  assertNoTurnIdMismatch,
  assertTurnStateAligned,
  eventsForFork,
  mkContextLimitHit,
  mkTurnOutcomeEventFailure,
  mkTurnOutcomeEventSuccess,
  mkTurnStarted,
} from './helpers'

describe('turn control context-limit path', () => {
  it.live('context_limit_hit then failed turn_outcome keeps same turnId', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-cl-1', chainId: 'c-cl' }))
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkTurnOutcomeEventFailure({ turnId: 't-cl-1', chainId: 'c-cl' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
      yield* assertTurnStateAligned(h)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('compaction in-progress path toggles blocking during context-limit hit', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'compaction_started', forkId: null, compactedMessageCount: 3 })
      yield* h.send(mkTurnStarted({ turnId: 't-cl-2', chainId: 'c-cl' }))
      yield* h.send(mkContextLimitHit(null, 'hard cap'))

      const blockedDuring = yield* h.projectionFork(CompactionProjection.Tag, null)
      expect(blockedDuring.contextLimitBlocked).toBe(true)

      yield* h.send(mkTurnOutcomeEventFailure({ turnId: 't-cl-2', chainId: 'c-cl' }))
      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('next turn after context-limit completion gets fresh turnId', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-cl-3a', chainId: 'c-cl-3' }))
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkTurnOutcomeEventFailure({ turnId: 't-cl-3a', chainId: 'c-cl-3' }))
      yield* h.send({ type: 'compaction_failed', forkId: null, error: 'cleanup' })

      yield* h.send(mkTurnStarted({ turnId: 't-cl-3b', chainId: 'c-cl-3' }))
      yield* h.send(mkTurnOutcomeEventSuccess({ turnId: 't-cl-3b', chainId: 'c-cl-3' }))

      const starts = eventsForFork(h, null).filter((e) => e.type === 'turn_started')
      expect(starts[0]?.turnId).not.toBe(starts[1]?.turnId)
      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('repeated context-limit cycles preserve ID pairing', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      for (const id of ['t-rep-1', 't-rep-2']) {
        yield* h.send(mkTurnStarted({ turnId: id, chainId: 'c-rep' }))
        yield* h.send(mkContextLimitHit(null, `ctx-${id}`))
        yield* h.send(mkTurnOutcomeEventFailure({ turnId: id, chainId: 'c-rep' }))
        yield* h.send({ type: 'compaction_failed', forkId: null, error: 'retry' })
      }

      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('full transcript invariant holds for context-limit sequence', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-a', chainId: 'c-full' }))
      yield* h.send(mkTurnOutcomeEventSuccess({ turnId: 't-a', chainId: 'c-full' }))
      yield* h.send(mkTurnStarted({ turnId: 't-b', chainId: 'c-full' }))
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkTurnOutcomeEventFailure({ turnId: 't-b', chainId: 'c-full' }))
      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 's',
        compactedMessageCount: 1,
        tokensSaved: 100,
        preservedVariables: [],
        refreshedContext: null,
      })
      yield* h.send(mkTurnStarted({ turnId: 't-c', chainId: 'c-full' }))
      yield* h.send(mkTurnOutcomeEventSuccess({ turnId: 't-c', chainId: 'c-full' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true } })))
  )
})
