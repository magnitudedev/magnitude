import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'

// Build XML strings from chars to avoid triggering the response parser
const WR = ['<','w','r','i','t','e',' ','p','a','t','h','='].join('')
const WRC = ['<','/','w','r','i','t','e','>'].join('')
const ACT_OPEN = ['<','a','c','t','i','o','n','s','>'].join('')
const ACT_CLOSE = ['<','/','a','c','t','i','o','n','s','>'].join('')
const YIELD = ['<','y','i','e','l','d','/','>'].join('')
const MSG_OPEN = ['<','c','o','m','m','s','>','<','m','e','s','s','a','g','e',' ','t','o','=','"','p','a','r','e','n','t','"','>'].join('')
const MSG_CLOSE = ['<','/','m','e','s','s','a','g','e','>','<','/','c','o','m','m','s','>'].join('')
const AC_OPEN = ['<','a','g','e','n','t','-','c','r','e','a','t','e',' '].join('')
const AC_CLOSE = ['<','/','a','g','e','n','t','-','c','r','e','a','t','e','>'].join('')

function writeTag(path: string, content: string) {
  return `${WR}"${path}">${content}${WRC}`
}

function agentCreate(agentId: string, type: string, title: string, message: string) {
  const typeTag = ['<','t','y','p','e','>'].join('') + type + ['<','/','t','y','p','e','>'].join('')
  const titleTag = ['<','t','i','t','l','e','>'].join('') + title + ['<','/','t','i','t','l','e','>'].join('')
  const msgTag = ['<','m','e','s','s','a','g','e','>'].join('') + message + ['<','/','m','e','s','s','a','g','e','>'].join('')
  return `${AC_OPEN}agentId="${agentId}" type="${type}">${typeTag}${titleTag}${msgTag}${AC_CLOSE}`
}

function actions(...tools: string[]) {
  return `${ACT_OPEN}${tools.join('')}${ACT_CLOSE}${YIELD}`
}

function actionsWithMessage(...tools: string[]) {
  return `${ACT_OPEN}${tools.join('')}${ACT_CLOSE}${MSG_OPEN}Done${MSG_CLOSE}${YIELD}`
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

      expect(completed.result.success).toBe(true)
      const writeCall = completed.toolCalls.find((c) => c.toolName === 'write')
      expect(writeCall).toBeDefined()
      expect(writeCall?.result.status).toBe('success')
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

      expect(completed.result.success).toBe(true)
      const writeCall = completed.toolCalls.find((c) => c.toolName === 'write')
      expect(writeCall).toBeDefined()
      expect(writeCall?.result.status).toBe('success')
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

      expect(completed.result.success).toBe(true)
      const writeCall = completed.toolCalls.find((c) => c.toolName === 'write')
      expect(writeCall).toBeDefined()
      expect(writeCall?.result.status).toBe('success')
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

      const writeCall = completed.toolCalls.find((c) => c.toolName === 'write')
      expect(writeCall).toBeDefined()
      // Should be rejected by policy
      expect(writeCall?.result.status).not.toBe('success')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
