import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { createAgentTestHarness } from '../harness'
import { MockTurnScriptTag } from '../turn-script'

describe('tool overrides', () => {
  test('Default shell returns empty success', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xml: '<actions><shell>echo hi</shell></actions><yield/>',
            },
            null,
          ),
        ),
      )

      await harness.user('run shell')
      const completed = await harness.wait.turnCompleted()

      expect(completed.result.success).toBe(true)
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      expect(shellCall?.result.status).toBe('success')
      if (shellCall?.result.status === 'success') {
        expect(shellCall.result.output).toEqual({ stdout: '', stderr: '', exitCode: 0 })
      }
    } finally {
      await harness.dispose()
    }
  })

  test('Custom toolOverride replaces default', async () => {
    const harness = await createAgentTestHarness({
      toolOverrides: {
        shell: () => ({ stdout: 'custom', stderr: '', exitCode: 42 }),
      },
    })

    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xml: '<actions><shell>echo hi</shell></actions><yield/>',
            },
            null,
          ),
        ),
      )

      await harness.user('run custom shell')
      const completed = await harness.wait.turnCompleted()

      expect(completed.result.success).toBe(true)
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      expect(shellCall?.result.status).toBe('success')
      if (shellCall?.result.status === 'success') {
        expect(shellCall.result.output).toEqual({ stdout: 'custom', stderr: '', exitCode: 42 })
      }
    } finally {
      await harness.dispose()
    }
  })

  test('Tool override that throws', async () => {
    const harness = await createAgentTestHarness({
      toolOverrides: {
        shell: () => {
          throw new Error('override failed')
        },
      },
    })

    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xml: '<actions><shell>echo hi</shell></actions><yield/>',
            },
            null,
          ),
        ),
      )

      await harness.user('run failing shell')
      const completed = await harness.wait.turnCompleted()

      expect(completed.type).toBe('turn_completed')
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      const hasToolError = shellCall?.result.status === 'error'
      const hasTurnError = completed.result.success === false

      expect(hasToolError || hasTurnError).toBe(true)

      if (shellCall?.result.status === 'error') {
        expect(shellCall.result.message).toContain('override failed')
      }
      if (completed.result.success === false) {
        expect(completed.result.error.length).toBeGreaterThan(0)
      }
    } finally {
      await harness.dispose()
    }
  })

  test('Async tool override', async () => {
    const harness = await createAgentTestHarness({
      toolOverrides: {
        shell: async () => ({ stdout: 'async', stderr: '', exitCode: 7 }),
      },
    })

    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xml: '<actions><shell>echo hi</shell></actions><yield/>',
            },
            null,
          ),
        ),
      )

      await harness.user('run async shell')
      const completed = await harness.wait.turnCompleted()

      expect(completed.result.success).toBe(true)
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      expect(shellCall?.result.status).toBe('success')
      if (shellCall?.result.status === 'success') {
        expect(shellCall.result.output).toEqual({ stdout: 'async', stderr: '', exitCode: 7 })
      }
    } finally {
      await harness.dispose()
    }
  })
})