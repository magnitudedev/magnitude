import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { response } from '../../src/test-harness/response-builder'

function writeTurn(path: string, content: string) {
  return response().writeFile(path, content).yield()
}

function agentCreateTurn(agentId: string, type: string, title: string, message: string) {
  return response().createAgent(agentId, type, title, message).yield()
}

function writeTurnWithParentMessage(path: string, content: string) {
  return {
    xml: `${response().messageTo('parent', 'Done').writeFile(path, content).yield().xml.replace('<magnitude:yield_user/>', '')}<magnitude:yield_user/>`,
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
      const writeEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.outcome._tag).toBe('Completed')
      if (writeEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(writeEnded.event.result._tag).toBe('Success')
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
      const writeEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.outcome._tag).toBe('Completed')
      if (writeEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(writeEnded.event.result._tag).toBe('Success')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('explorer can write to $M/', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness

      // Lead spawns an explorer
      yield* harness.script.next({
        ...agentCreateTurn('explorer-test', 'explorer', 'Test', 'Write a report'),
      }, null)

      yield* harness.user('explore and write report')

      const agentCreated = yield* harness.wait.agentCreated(
        (e) => e.agentId === 'explorer-test',
      )

      // Explorer writes to $M/
      yield* harness.script.next({
        ...writeTurnWithParentMessage('$M/reports/test.md', 'hello from explorer'),
      }, agentCreated.forkId)

      const completed = yield* harness.wait.turnCompleted(agentCreated.forkId)
      const writeEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === agentCreated.forkId && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.outcome._tag).toBe('Completed')
      if (writeEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(writeEnded.event.result._tag).toBe('Success')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('explorer cannot write to cwd', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness

      // Lead spawns an explorer
      yield* harness.script.next({
        ...agentCreateTurn('explorer-cwd', 'explorer', 'Test', 'Try writing to cwd'),
      }, null)

      yield* harness.user('explore')

      const agentCreated = yield* harness.wait.agentCreated(
        (e) => e.agentId === 'explorer-cwd',
      )

      // Explorer tries to write to cwd - should be denied
      yield* harness.script.next({
        ...writeTurnWithParentMessage('src/test.txt', 'hello from explorer'),
      }, agentCreated.forkId)

      const completed = yield* harness.wait.turnCompleted(agentCreated.forkId)
      const writeEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === agentCreated.forkId && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      if (writeEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(writeEnded.event.result._tag).not.toBe('Success')
      if (completed.outcome._tag === 'Completed') {
        expect(completed.outcome.completion.yieldTarget).toBe('invoke')
      } else {
        expect(completed.outcome._tag).not.toBe('Cancelled')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
