import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'

// Build XML strings from chars to avoid triggering the response parser
const WR = ['<','w','r','i','t','e',' ','p','a','t','h','='].join('')
const WRC = ['<','/','w','r','i','t','e','>'].join('')
const YIELD = ['<','i','d','l','e','/','>'].join('')
const MSG_OPEN = ['<','m','e','s','s','a','g','e',' ','t','o','=','"','p','a','r','e','n','t','"','>'].join('')
const MSG_CLOSE = ['<','/','m','e','s','s','a','g','e','>'].join('')
const AC_OPEN = ['<','a','g','e','n','t','-','c','r','e','a','t','e',' '].join('')
const AC_CLOSE = ['<','/','a','g','e','n','t','-','c','r','e','a','t','e','>'].join('')

function writeTag(path: string, content: string) {
  return `${WR}"${path}">${content}${WRC}`
}

function agentCreate(agentId: string, type: string, title: string, message: string) {
  const titleTag = ['<','t','i','t','l','e','>'].join('') + title + ['<','/','t','i','t','l','e','>'].join('')
  const msgTag = ['<','m','e','s','s','a','g','e','>'].join('') + message + ['<','/','m','e','s','s','a','g','e','>'].join('')
  return `${AC_OPEN}id="${agentId}" type="${type}">${titleTag}${msgTag}${AC_CLOSE}`
}

function actions(...tools: string[]) {
  return `${tools.join('')}${YIELD}`
}

function actionsWithMessage(...tools: string[]) {
  return `${tools.join('')}${MSG_OPEN}Done${MSG_CLOSE}${YIELD}`
}

describe('write policy permissions', () => {
  it.live('lead can write to $M/', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({
        xml: actions(writeTag('$M/reports/test.md', 'hello')),
      }, null)

      yield* harness.user('write a report')
      const completed = yield* harness.wait.turnCompleted(null)
      const writeEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.result.success).toBe(true)
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
        xml: actions(writeTag('src/test.txt', 'hello')),
      }, null)

      yield* harness.user('write a file')
      const completed = yield* harness.wait.turnCompleted(null)
      const writeEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.result.success).toBe(true)
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
        xml: actions(agentCreate('explorer-test', 'explorer', 'Test', 'Write a report')),
      }, null)

      yield* harness.user('explore and write report')

      const agentCreated = yield* harness.wait.agentCreated(
        (e) => e.agentId === 'explorer-test',
      )

      // Explorer writes to $M/
      yield* harness.script.next({
        xml: actionsWithMessage(writeTag('$M/reports/test.md', 'hello from explorer')),
      }, agentCreated.forkId)

      const completed = yield* harness.wait.turnCompleted(agentCreated.forkId)
      const writeEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === agentCreated.forkId && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.result.success).toBe(true)
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
        xml: actions(agentCreate('explorer-cwd', 'explorer', 'Test', 'Try writing to cwd')),
      }, null)

      yield* harness.user('explore')

      const agentCreated = yield* harness.wait.agentCreated(
        (e) => e.agentId === 'explorer-cwd',
      )

      // Explorer tries to write to cwd - should be denied
      yield* harness.script.next({
        xml: actionsWithMessage(writeTag('src/test.txt', 'hello from explorer')),
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
      if (completed.result.success) {
        expect(completed.result.turnDecision).toBe('continue')
      } else {
        expect(completed.result.cancelled).toBe(false)
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
