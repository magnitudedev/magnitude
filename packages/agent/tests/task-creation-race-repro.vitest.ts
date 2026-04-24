import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { response } from '../src/test-harness/response-builder'

describe('task creation + spawn-worker', () => {
  it.live('create-task + spawn-worker in same turn', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Turn 1: create task AND spawn worker in the SAME turn
      yield* h.script.next(
        response()
          .createTask('my-task', 'My task')
          .spawnWorker('my-task', 'Do the work')
          .message('Created task and spawned worker.')
          .yield()
      )

      yield* h.user('test')

      const rootTurn = yield* h.wait.turnCompleted(null)
      
      // Turn should succeed (idle, not continue which means errors)
      expect(rootTurn.result._tag).toBe('Completed')
      expect(rootTurn.result.completion.decision).toBe('idle')

      // spawn-worker should create the agent
      const agentCreated = yield* h.wait.event('agent_created' as any, (e: any) => e.agentId === 'my-task', { timeoutMs: 5000 })
      expect(agentCreated).toBeDefined()
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
