import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { response, TestHarness, TestHarnessLive } from '../test-harness/harness'

describe('task creation race condition repro', () => {
  it.live('create_task + spawn_worker in same turn: spawn_worker fails because projection not updated', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Turn 1: create task AND spawn worker in the SAME turn
      // This is what the model does: create_task then spawn_worker back to back
      yield* h.script.next(
        response()
          .createTask('my-task', 'builder', 'My task')
          .spawnWorker('my-task', 'builder', 'Do the work')
          .message('Created task and spawned worker.')
          .yield()
      )

      yield* h.user('test')

      const rootTurn = yield* h.wait.turnCompleted(null)
      console.log('root turn result:', JSON.stringify(rootTurn.outcome))

      // Check if task was created
      const taskCreated = yield* h.wait.event('task_created', (e) => e.taskId === 'my-task')
      console.log('task_created event:', JSON.stringify(taskCreated))

      // Check if spawn_worker failed
      const toolEvents = h.transcript.filter(e => e.type === 'tool_event')
      console.log('tool events:')
      for (const e of toolEvents) {
        console.log(`  ${e.event._tag} ${e.event.toolName} toolCallId=${e.event.toolCallId}`)
        if (e.event._tag === 'ToolExecutionEnded') {
          console.log(`    result: ${JSON.stringify(e.event.result)}`)
        }
        if (e.event._tag === 'ToolParseError' || e.event._tag === 'StructuralParseError') {
          console.log(`    error: ${JSON.stringify(e.event.error)}`)
        }
      }

      // This should pass but WILL FAIL if spawn_worker can't find the task
      const agentCreated = yield* h.wait.event('agent_created', (e) => e.agentId === 'my-task')
      expect(agentCreated).toBeDefined()
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
