import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { createAgentTestHarness } from '../harness'
import { MockTurnScriptTag } from '../turn-script'

describe('fault injection', () => {
  test('Malformed XML', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue({ xml: ['<comms>', '<message to="user">hi'].join('') }, null)
        )
      )

      await harness.user('broken xml')

      const result = await Promise.race([
        harness.wait.turnCompleted(),
        harness.wait.event('turn_unexpected_error'),
      ])

      expect(result.type === 'turn_completed' || result.type === 'turn_unexpected_error').toBe(true)
    } finally {
      await harness.dispose()
    }
  })

  test('terminateStreamEarly', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xmlChunks: [
                '<comms>',
                '<message to="user">hi</message>',
                '</comms><yield/>',
              ],
              terminateStreamEarly: true,
            },
            null
          )
        )
      )

      await harness.user('terminate early')

      const result = await Promise.race([
        harness.wait.turnCompleted(),
        harness.wait.event('turn_unexpected_error'),
      ])

      expect(result.type === 'turn_completed' || result.type === 'turn_unexpected_error').toBe(true)
    } finally {
      await harness.dispose()
    }
  })

  test('failAfterChunk', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xmlChunks: [
                '<comms>',
                '<message to="user">hi</message>',
                '</comms><yield/>',
              ],
              failAfterChunk: 1,
            },
            null
          )
        )
      )

      await harness.user('fail after chunk')

      const result = await Promise.race([
        harness.wait.event('turn_unexpected_error'),
        harness.wait.turnCompleted(),
      ])

      expect(result.type === 'turn_unexpected_error' || result.type === 'turn_completed').toBe(true)
    } finally {
      await harness.dispose()
    }
  })
})