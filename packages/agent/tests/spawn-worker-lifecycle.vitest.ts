import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'

describe('spawn-worker lifecycle integration', () => {
  it.live('create-task -> spawn-worker -> message -> worker executes tool -> worker responds', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Turn 1 (root): create task
      yield* h.script.next({
        xml: '<invoke tool="create_task">\n<parameter name="id">flow-task</parameter>\n<parameter name="title">Flow task</parameter>\n</invoke>\n<yield_user/>',
      }, null)

      // Turn 2 (root): spawn worker with initial instructions
      yield* h.script.next({
        xml: '<invoke tool="spawn_worker">\n<parameter name="id">flow-task</parameter>\n<parameter name="message">Write output.txt with content "hello"</parameter>\n</invoke>\n<yield_user/>',
      }, null)

      // Turn 3 (worker): execute tool + respond to parent
      yield* h.script.next({
        xml: '<invoke tool="write">\n<parameter name="path">output.txt</parameter>\n<parameter name="content">hello</parameter>\n</invoke>\n<message to="parent">done</message>\n<yield_parent/>',
      })

      // Turn 4 (root): follow-up to user
      yield* h.script.next({
        xml: '<message to="user">received</message>\n<yield_user/>',
      }, null)

      yield* h.user('run spawn-worker lifecycle flow')

      const rootFirst = yield* h.wait.turnCompleted(null)
      expect(rootFirst.result.success).toBe(true)

      // Trigger second root turn to run spawn-worker frame
      yield* h.user('continue')

      const rootSecond = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== rootFirst.turnId,
      )
      expect(rootSecond.result.success).toBe(true)

      const created = yield* h.wait.event('agent_created', (e) => e.agentId === 'flow-task')
      const workerCompleted = yield* h.wait.turnCompleted(created.forkId)
      const workerWrite = yield* h.wait.event(
        'tool_event',
        (e) => e.forkId === created.forkId && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )
      expect(workerCompleted.result.success).toBe(true)
      if (workerWrite.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(workerWrite.event.result._tag).toBe('Success')

      const rootFollowUp = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== rootFirst.turnId && e.turnId !== rootSecond.turnId,
      )
      expect(rootFollowUp.result.success).toBe(true)

      expect(h.files.get('output.txt')).toBe('hello')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
