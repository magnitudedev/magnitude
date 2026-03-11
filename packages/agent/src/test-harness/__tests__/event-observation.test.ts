import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { createAgentTestHarness } from '../harness'
import { MockTurnScriptTag } from '../turn-script'

const simpleYieldXml = [
  '<comms>',
  '<message to="user">hi</message>',
  '</comms>',
  '<yield/>',
].join('')

describe('event observation', () => {
  test('events() captures all events', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: simpleYieldXml }, null)
        )
      )

      await harness.user('hello')
      await harness.wait.turnCompleted()

      const types = harness.events().map((e) => e.type)
      expect(types).toContain('session_initialized')
      expect(types).toContain('user_message')
      expect(types).toContain('turn_started')
      expect(types).toContain('turn_completed')
    } finally {
      await harness.dispose()
    }
  })

  test('wait.event() resolves on match', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: simpleYieldXml }, null)
        )
      )

      await harness.user('trigger')
      const started = await harness.wait.event('turn_started')

      expect(started.type).toBe('turn_started')
      expect(started.forkId).toBeNull()
    } finally {
      await harness.dispose()
    }
  })

  test('wait.turnCompleted() with forkId', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: simpleYieldXml }, null)
        )
      )

      await harness.user('root turn')
      const completed = await harness.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.forkId).toBeNull()
    } finally {
      await harness.dispose()
    }
  })

  test('onEvent callback fires', async () => {
    const harness = await createAgentTestHarness()
    try {
      const seen: string[] = []
      const unsubscribe = harness.onEvent((e) => {
        seen.push(e.type)
      })

      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: simpleYieldXml }, null)
        )
      )

      await harness.user('hello')
      await harness.wait.turnCompleted()
      unsubscribe()

      expect(seen.length).toBeGreaterThan(0)
      expect(seen).toContain('user_message')
      expect(seen).toContain('turn_started')
      expect(seen).toContain('turn_completed')
    } finally {
      await harness.dispose()
    }
  })

  test('Direct event injection', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.send({
        type: 'user_message',
        forkId: null,
        content: [],
        attachments: [],
        mode: 'text',
        synthetic: true,
        taskMode: false,
      })

      const transcript = harness.events()
      const hasInjected = transcript.some((e) => e.type === 'user_message' && e.synthetic === true)
      expect(hasInjected).toBe(true)
    } finally {
      await harness.dispose()
    }
  })
})