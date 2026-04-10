import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'

describe('harness shell behavior', () => {
  it.live('shell executes successfully', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: '<shell>echo hi</shell><idle/>' }, null)

      yield* harness.user('run shell')
      const completed = yield* harness.wait.turnCompleted(null)
      const toolEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'shell',
      )

      expect(completed.result.success).toBe(true)
      if (toolEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(toolEnded.event.result._tag).toBe('Success')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
