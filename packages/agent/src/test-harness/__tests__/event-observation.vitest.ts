import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'
import { textParts } from '../../content'
import { createId } from '../../util/id'

const simpleYieldXml = '<comms><message to="user">hi</message></comms><yield/>'

describe('event observation', () => {
  it.live('events() captures all events', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: simpleYieldXml }, null)
      yield* harness.user('hello')
      yield* harness.wait.turnCompleted(null)

      const types = harness.events().map((e) => e.type)
      expect(types).toContain('session_initialized')
      expect(types).toContain('user_message')
      expect(types).toContain('turn_started')
      expect(types).toContain('turn_completed')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('wait.event() resolves on match', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: simpleYieldXml }, null)

      yield* harness.user('trigger')
      const started = yield* harness.wait.event('turn_started')

      expect(started.type).toBe('turn_started')
      expect(started.forkId).toBeNull()
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('wait.turnCompleted() with forkId', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: simpleYieldXml }, null)

      yield* harness.user('root turn')
      const completed = yield* harness.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.forkId).toBeNull()
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Direct event injection via send()', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness

      yield* harness.send({
        type: 'user_message',
        messageId: createId(),
        forkId: null,
        timestamp: Date.now(),
        content: textParts('injected'),
        attachments: [],
        mode: 'text',
        synthetic: true,
        taskMode: false,
      })

      const transcript = harness.events()
      const hasInjected = transcript.some((e) => e.type === 'user_message' && e.synthetic === true)
      expect(hasInjected).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
