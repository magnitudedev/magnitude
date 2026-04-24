import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'

describe('harness shell behavior', () => {
  it.live('shell executes successfully', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hi</magnitude:parameter>\n</magnitude:invoke><magnitude:yield_user/>' }, null)

      yield* harness.user('run shell')
      const completed = yield* harness.wait.turnCompleted(null)
      const toolEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'shell',
      )

      expect(completed.result._tag).toBe('Completed')
      if (toolEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(toolEnded.event.result._tag).toBe('Success')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
