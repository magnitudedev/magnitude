import { describe, expect, it } from '@effect/vitest'
import { Effect, Layer, Ref } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { response } from '../src/test-harness/response-builder'

describe('real tool execution repro', () => {
  it.live('create-task + spawn-worker with real tools', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Turn 1: create task AND spawn worker in same turn
      yield* h.script.next(
        response()
          .createTask('my-task', 'builder', 'My task')
          .spawnWorker('my-task', 'builder', 'Do the work')
          .message('Created task and spawned worker.')
          .yield()
      )

      yield* h.user('test')

      // Wait for root turn to complete
      const rootTurn = yield* h.wait.turnCompleted(null)
      
      console.log('root turn result:', JSON.stringify(rootTurn.result))

      // Check if create-task tool actually executed
      const createTaskExec = yield* h.wait.event('tool_event' as any, (e: any) =>
        e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'create-task'
      , { timeoutMs: 3000 })
      console.log('create-task result:', JSON.stringify(createTaskExec.event.result))

      // Check if spawn-worker tool executed
      const spawnWorkerExec = yield* h.wait.event('tool_event' as any, (e: any) =>
        e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'spawn-worker'
      , { timeoutMs: 3000 }).pipe(
        Effect.catchAll(() => Effect.succeed(null))
      )
      
      if (spawnWorkerExec) {
        console.log('spawn-worker result:', JSON.stringify((spawnWorkerExec as any).event.result))
      } else {
        console.log('spawn-worker: NO ToolExecutionEnded event (tool never executed or failed before execution)')
        
        // Check for parse error
        const parseError = yield* h.wait.event('tool_event' as any, (e: any) =>
          e.event._tag === 'ToolParseError' && e.event.toolName === 'spawn-worker'
        , { timeoutMs: 1000 }).pipe(
          Effect.catchAll(() => Effect.succeed(null))
        )
        if (parseError) {
          console.log('spawn-worker PARSE ERROR:', JSON.stringify((parseError as any).event.error))
        }
      }

      // The definitive assertion: does agent_created happen?
      const agentCreated = yield* h.wait.event('agent_created' as any, (e: any) => e.agentId === 'my-task', { timeoutMs: 5000 }).pipe(
        Effect.catchAll(() => Effect.succeed(null))
      )
      
      expect(agentCreated).not.toBeNull()
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
