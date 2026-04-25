import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { assertPrefixUnchanged, getRootMemory, sendUserMessage, snapshotMessageRefs } from './helpers'

function sendUserMessageReady(h: Effect.Effect.Success<typeof TestHarness>, text: string, timestamp: number) {
  return sendUserMessage(h, { text, timestamp })
}

describe('memory/append-only-invariant', () => {
  it.live('after user_message_ready prior messages remain unchanged', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* sendUserMessageReady(h, 'baseline', 1_710_000_010_000)

      const before = yield* getRootMemory(h)
      const snap = snapshotMessageRefs(before)

      yield* sendUserMessageReady(h, 'next message', 1_710_000_010_001)

      const after = yield* getRootMemory(h)
      expect(after.messages.length).toBeGreaterThanOrEqual(snap.refs.length)
      assertPrefixUnchanged(snap, after)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('after observations_captured prior messages remain unchanged', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessageReady(h, 'before observation', 1_710_000_010_010)
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 'turn-obs', chainId: 'chain-1' })

      const before = yield* getRootMemory(h)
      const snap = snapshotMessageRefs(before)

      yield* h.send({
        type: 'observations_captured',
        forkId: null,
        turnId: 'turn-obs',
        parts: [{ type: 'text', text: 'observation text' }],
      })

      const after = yield* getRootMemory(h)
      expect(after.messages.length).toBe(snap.refs.length + 1)
      assertPrefixUnchanged(snap, after)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('after turn_outcome prior messages remain unchanged', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessageReady(h, 'before error', 1_710_000_010_020)
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 'turn-err', chainId: 'chain-2' })

      const before = yield* getRootMemory(h)
      const snap = snapshotMessageRefs(before)

      yield* h.send({
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-err',
        message: 'boom',
      })

      const after = yield* getRootMemory(h)
      expect(after.messages.length).toBe(snap.refs.length + 1)
      assertPrefixUnchanged(snap, after)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('no message mutation across mixed event sequence', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessageReady(h, 'mixed start', 1_710_000_010_030)
      let state = yield* getRootMemory(h)
      let snap = snapshotMessageRefs(state)

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 'turn-mixed', chainId: 'chain-3' })
      state = yield* getRootMemory(h)
      expect(state.messages.length).toBe(snap.refs.length + 1)
      assertPrefixUnchanged(snap, state)
      snap = snapshotMessageRefs(state)

      yield* sendUserMessageReady(h, 'queued during turn', 1_710_000_010_031)
      state = yield* getRootMemory(h)
      expect(state.messages.length).toBe(snap.refs.length)
      assertPrefixUnchanged(snap, state)

      yield* h.send({
        type: 'observations_captured',
        forkId: null,
        turnId: 'turn-mixed',
        parts: [{ type: 'text', text: 'obs in mixed flow' }],
      })
      state = yield* getRootMemory(h)
      expect(state.messages.length).toBe(snap.refs.length + 1)
      assertPrefixUnchanged(snap, state)
      snap = snapshotMessageRefs(state)

      yield* h.send({
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-mixed',
        message: 'mixed error',
      })
      state = yield* getRootMemory(h)
      expect(state.messages.length).toBeGreaterThan(snap.refs.length)
      assertPrefixUnchanged(snap, state)
      snap = snapshotMessageRefs(state)

      yield* h.send({
        type: 'interrupt',
        forkId: null,
      })
      state = yield* getRootMemory(h)
      expect(state.messages.length).toBeGreaterThanOrEqual(snap.refs.length)
      assertPrefixUnchanged(snap, state)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('reference snapshot invariant helper catches accidental merge', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessageReady(h, 'msg1', 1_710_000_010_040)
      let before = yield* getRootMemory(h)
      let snap = snapshotMessageRefs(before)

      yield* sendUserMessageReady(h, 'msg2', 1_710_000_010_041)
      let after = yield* getRootMemory(h)
      assertPrefixUnchanged(snap, after)

      snap = snapshotMessageRefs(after)
      yield* sendUserMessageReady(h, 'msg3', 1_710_000_010_042)
      after = yield* getRootMemory(h)
      assertPrefixUnchanged(snap, after)

      snap = snapshotMessageRefs(after)
      yield* sendUserMessageReady(h, 'msg4', 1_710_000_010_043)
      after = yield* getRootMemory(h)
      assertPrefixUnchanged(snap, after)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )
})
