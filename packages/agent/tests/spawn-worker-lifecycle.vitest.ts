import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'

describe('spawn-worker lifecycle integration', () => {
  it.live('create-task -> spawn-worker -> message -> worker executes tool -> worker responds', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Turn 1 (root): create task
      yield* h.script.next({
        xml: '<magnitude:invoke tool="create_task">\n<magnitude:parameter name="id">flow-task</magnitude:parameter>\n<magnitude:parameter name="title">Flow task</magnitude:parameter>\n</magnitude:invoke>\n<magnitude:yield_user/>',
      }, null)

      // Turn 2 (root): spawn worker with initial instructions
      yield* h.script.next({
        xml: '<magnitude:invoke tool="spawn_worker">\n<magnitude:parameter name="id">flow-task</magnitude:parameter>\n<magnitude:parameter name="message">Write output.txt with content "hello"</magnitude:parameter>\n</magnitude:invoke>\n<magnitude:yield_user/>',
      }, null)

      // Turn 3 (worker): execute tool + respond to parent
      yield* h.script.next({
        xml: '<magnitude:invoke tool="write">\n<magnitude:parameter name="path">output.txt</magnitude:parameter>\n<magnitude:parameter name="content">hello</magnitude:parameter>\n</magnitude:invoke>\n<magnitude:message to="parent">done</magnitude:message>\n<magnitude:yield_parent/>',
      })

      // Turn 4 (root): follow-up to user
      yield* h.script.next({
        xml: '<magnitude:message to="user">received</magnitude:message>\n<magnitude:yield_user/>',
      }, null)

      yield* h.user('run spawn-worker lifecycle flow')

      const rootFirst = yield* h.wait.turnCompleted(null)
      expect(rootFirst.outcome._tag).toBe('Completed')

      // Trigger second root turn to run spawn-worker frame
      yield* h.user('continue')

      const rootSecond = yield* h.wait.event(
        'turn_outcome',
        (e) => e.forkId === null && e.turnId !== rootFirst.turnId,
      )
      expect(rootSecond.outcome._tag).toBe('Completed')

      const created = yield* h.wait.event('agent_created', (e) => e.agentId === 'flow-task')
      const workerCompleted = yield* h.wait.turnCompleted(created.forkId)
      const workerWrite = yield* h.wait.event(
        'tool_event',
        (e) =>
          e.forkId === created.forkId
          && e.event._tag === 'ToolExecutionEnded'
          && e.event.toolName === 'write',
      )
      expect(workerCompleted.outcome._tag).toBe('Completed')
      expect(workerWrite.event._tag === 'ToolExecutionEnded' && workerWrite.event.result._tag).toBe('Success')

      const rootFollowUp = yield* h.wait.event(
        'turn_outcome',
        (e) => e.forkId === null && e.turnId !== rootFirst.turnId && e.turnId !== rootSecond.turnId,
      )
      expect(rootFollowUp.outcome._tag).toBe('Completed')

      expect(h.files.get('output.txt')).toBe('hello')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
