import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'

describe('harness shell behavior', () => {
  it.live('shell executes successfully', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: '<actions><shell>echo hi</shell></actions><idle/>' }, null)

      yield* harness.user('run shell')
      const completed = yield* harness.wait.turnCompleted(null)

      expect(completed.result.success).toBe(true)
      const shellCall = completed.toolCalls.find((c) => c.toolName === 'shell')
      expect(shellCall?.result.status).toBe('success')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
