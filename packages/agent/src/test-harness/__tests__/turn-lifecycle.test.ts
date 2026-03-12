import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { createAgentTestHarness } from '../harness'
import { MockTurnScriptTag } from '../turn-script'

describe('turn lifecycle', () => {
  test('Single turn with yield', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: '<comms><message to="user">hi</message></comms><yield/>' }, null)
        )
      )

      await harness.user('hello')
      const completed = await harness.wait.turnCompleted()

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)
      if (completed.result.success) {
        expect(completed.result.turnDecision).toBe('yield')
      }
    } finally {
      await harness.dispose()
    }
  })

  test('Single turn with next', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: '<comms><message to="user">first</message></comms><next/>' }, null)
        )
      )

      await harness.user('run')
      const completed = await harness.wait.turnCompleted()

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)
    } finally {
      await harness.dispose()
    }
  })

  test('Multi-turn conversation', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: '<comms><message to="user">response 1</message></comms><yield/>' }, null)
        )
      )
      await harness.user('message 1')
      const first = await harness.wait.turnCompleted()

      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: '<comms><message to="user">response 2</message></comms><yield/>' }, null)
        )
      )
      await harness.user('message 2')
      const second = await harness.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId
      )

      expect(first.type).toBe('turn_completed')
      expect(second.type).toBe('turn_completed')
      expect(first.result.success).toBe(true)
      expect(second.result.success).toBe(true)
    } finally {
      await harness.dispose()
    }
  })

  test('Default frame when no script queued', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.user('no script queued')
      const completed = await harness.wait.turnCompleted()

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)

      const text = completed.responseParts.find((p) => p.type === 'text')
      expect(text?.type).toBe('text')
      if (text?.type === 'text') {
        expect(text.content).toContain('ok')
      }
    } finally {
      await harness.dispose()
    }
  })
})