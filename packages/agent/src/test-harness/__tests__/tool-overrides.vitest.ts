import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'

const shellTurn = '<actions><shell>echo hi</shell></actions><yield/>'

describe('tool overrides', () => {
  it.live('Default shell returns empty success', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: shellTurn }, null)

      yield* harness.user('run shell')
      const completed = yield* harness.wait.turnCompleted(null)

      expect(completed.result.success).toBe(true)
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      expect(shellCall?.result.status).toBe('success')
      if (shellCall?.result.status === 'success') {
        expect(shellCall.result.output).toEqual({ stdout: '', stderr: '', exitCode: 0 })
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Custom toolOverride replaces default', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: shellTurn }, null)

      yield* harness.user('run custom shell')
      const completed = yield* harness.wait.turnCompleted(null)

      expect(completed.result.success).toBe(true)
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      expect(shellCall?.result.status).toBe('success')
      if (shellCall?.result.status === 'success') {
        expect(shellCall.result.output).toEqual({ stdout: 'custom', stderr: '', exitCode: 42 })
      }
    }).pipe(
      Effect.provide(
        TestHarnessLive({
          toolOverrides: {
            shell: () => ({ stdout: 'custom', stderr: '', exitCode: 42 }),
          },
        }),
      ),
    )
  )

  it.live('Tool override that throws', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: shellTurn }, null)

      yield* harness.user('run failing shell')
      const completed = yield* harness.wait.turnCompleted(null)

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
    }).pipe(
      Effect.provide(
        TestHarnessLive({
          toolOverrides: {
            shell: () => {
              throw new Error('override failed')
            },
          },
        }),
      ),
    )
  )

  it.live('Async tool override', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: shellTurn }, null)

      yield* harness.user('run async shell')
      const completed = yield* harness.wait.turnCompleted(null)

      expect(completed.result.success).toBe(true)
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      expect(shellCall?.result.status).toBe('success')
      if (shellCall?.result.status === 'success') {
        expect(shellCall.result.output).toEqual({ stdout: 'async', stderr: '', exitCode: 7 })
      }
    }).pipe(
      Effect.provide(
        TestHarnessLive({
          toolOverrides: {
            shell: async () => ({ stdout: 'async', stderr: '', exitCode: 7 }),
          },
        }),
      ),
    )
  )
})
