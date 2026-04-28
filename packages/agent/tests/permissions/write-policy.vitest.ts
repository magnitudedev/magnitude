import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { response } from '../../src/test-harness/response-builder'

// Helper: tool_event predicate matching ToolExecutionEnded for a given fork+tool name.
const writeEndedFor = (forkId: string | null) =>
  (e: { readonly forkId: string | null; readonly event: { readonly _tag: string; readonly toolName?: string } }) =>
    e.forkId === forkId
    && e.event._tag === 'ToolExecutionEnded'
    && e.event.toolName === 'write'

const isSuccess = (e: { readonly event: unknown }): boolean => {
  const ev = e.event as { _tag?: string; result?: { _tag?: string } }
  return ev._tag === 'ToolExecutionEnded' && ev.result?._tag === 'Success'
}

function writeTurn(path: string, content: string) {
  return response().writeFile(path, content).yield()
}

function agentCreateTurn(agentId: string, _type: string, title: string, message: string) {
  // Lead variant only spawns 'worker' agents now. Create a task and spawn a
  // worker for it; the resulting agent's agentId equals the task id.
  return response()
    .createTask(agentId, title)
    .spawnWorker(agentId, message)
    .yield()
}

function writeTurnWithParentMessage(path: string, content: string) {
  return {
    xml: `${response().messageTo('parent', 'Done').writeFile(path, content).yield().xml!.replace('<magnitude:yield_user/>', '')}<magnitude:yield_user/>`,
  }
}

describe('write policy permissions', () => {
  it.live('lead can write to $M/', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({
        ...writeTurn('$M/reports/test.md', 'hello'),
      }, null)

      yield* harness.user('write a report')
      const completed = yield* harness.wait.turnCompleted(null)
      const writeEnded = yield* harness.wait.event('tool_event', writeEndedFor(null))

      expect(completed.outcome._tag).toBe('Completed')
      expect(isSuccess(writeEnded)).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('lead can write to cwd', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({
        ...writeTurn('src/test.txt', 'hello'),
      }, null)

      yield* harness.user('write a file')
      const completed = yield* harness.wait.turnCompleted(null)
      const writeEnded = yield* harness.wait.event('tool_event', writeEndedFor(null))

      expect(completed.outcome._tag).toBe('Completed')
      expect(isSuccess(writeEnded)).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('worker can write to $M/', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness

      // Lead spawns an explorer
      yield* harness.script.next({
        ...agentCreateTurn('worker-test', 'worker', 'Test', 'Write a report'),
      }, null)

      yield* harness.user('explore and write report')

      const agentCreated = yield* harness.wait.agentCreated(
        (e) => e.agentId === 'worker-test',
      )

      // Worker writes to $M/
      yield* harness.script.next({
        ...writeTurnWithParentMessage('$M/reports/test.md', 'hello from worker'),
      }, agentCreated.forkId)

      const completed = yield* harness.wait.turnCompleted(agentCreated.forkId)
      const writeEnded = yield* harness.wait.event('tool_event', writeEndedFor(agentCreated.forkId))

      expect(completed.outcome._tag).toBe('Completed')
      expect(isSuccess(writeEnded)).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('worker can write to cwd', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness

      // Lead spawns a worker
      yield* harness.script.next({
        ...agentCreateTurn('worker-cwd', 'worker', 'Test', 'Try writing to cwd'),
      }, null)

      yield* harness.user('explore')

      const agentCreated = yield* harness.wait.agentCreated(
        (e) => e.agentId === 'worker-cwd',
      )

      // Workers share lead's write policy and can write inside cwd.
      yield* harness.script.next({
        ...writeTurnWithParentMessage('src/test.txt', 'hello from worker'),
      }, agentCreated.forkId)

      const completed = yield* harness.wait.turnCompleted(agentCreated.forkId)
      const writeEnded = yield* harness.wait.event('tool_event', writeEndedFor(agentCreated.forkId))

      expect(completed.outcome._tag).toBe('Completed')
      expect(isSuccess(writeEnded)).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
