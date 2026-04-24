import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'
import { MockTurnScriptTag } from '../turn-script'
import { TurnProjection } from '../../projections/turn'

describe('baseline harness lifecycle', () => {
  it.live('1) initializes cleanly', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      expect(Array.isArray(harness.events())).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('2) user message -> single turn -> yield -> idle', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({
        xml: '<magnitude:message to="user">done</magnitude:message><magnitude:yield_user/>',
      })

      yield* harness.user('hello')
      const completed = yield* harness.wait.turnCompleted(null)

      expect(completed.result._tag).toBe('Completed')
      if (completed.result._tag === 'Completed') {
        expect(completed.result.completion.decision).toBe('idle')
      }

      const root = yield* harness.projectionFork(TurnProjection.Tag, null)
      expect(root._tag).toBe('idle')
      expect(root.triggers.length).toBe(0)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('3) multi-turn chain (next then yield)', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo chain</magnitude:parameter>\n</magnitude:invoke>' })
      yield* harness.script.next({ xml: '<magnitude:yield_user/>' })

      yield* harness.user('run chain')

      const first = yield* harness.wait.turnCompleted(null)
      const second = yield* harness.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId,
      )

      expect(first.result._tag).toBe('Completed')
      if (first.result._tag === 'Completed') {
        expect(first.result.completion.decision).toBe('continue')
      }

      expect(second.result._tag).toBe('Completed')
      if (second.result._tag === 'Completed') {
        expect(second.result.completion.decision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('4) tool execution in a turn (write)', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({
        xml: '<magnitude:invoke tool="write">\n<magnitude:parameter name="path">output.txt</magnitude:parameter>\n<magnitude:parameter name="content">content</magnitude:parameter>\n</magnitude:invoke><magnitude:yield_user/>',
      })

      yield* harness.user('write a file')
      const completed = yield* harness.wait.turnCompleted(null)
      const toolEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.result._tag).toBe('Completed')
      if (toolEnded.event._tag !== 'ToolExecutionEnded') {
        throw new Error('Expected ToolExecutionEnded')
      }
      expect(toolEnded.event.result._tag).toBe('Success')
      expect(harness.files.get('output.txt')).toBe('content')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('5) subagent message to parent triggers parent turn', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      let rootTurns = 0

      yield* harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (script) =>
          script.setResolver(({ forkId }) => {
            if (forkId === null) {
              rootTurns += 1
              if (rootTurns === 1) {
                return {
                  xml: '<magnitude:invoke tool="agent_create">\n<magnitude:parameter name="agentId">baseline-sub</magnitude:parameter>\n<magnitude:parameter name="type">explorer</magnitude:parameter>\n<magnitude:parameter name="title">baseline</magnitude:parameter>\n<magnitude:parameter name="message">do work</magnitude:parameter>\n</magnitude:invoke><magnitude:yield_user/>',
                }
              }
              return { xml: '<magnitude:yield_user/>' }
            }

            return { xml: '<magnitude:message to="parent">subagent done</magnitude:message><magnitude:yield_user/>' }
          }),
        ),
      )

      yield* harness.user('start subagent flow')

      const rootFirst = yield* harness.wait.turnCompleted(null)
      expect(rootFirst.result._tag).toBe('Completed')

      const created = yield* harness.wait.agentCreated((e) => e.agentId === 'baseline-sub')
      expect(created.forkId).not.toBeNull()

      const subCompleted = yield* harness.wait.turnCompleted(created.forkId)
      expect(subCompleted.result._tag).toBe('Completed')

      const rootSecond = yield* harness.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== rootFirst.turnId,
      )
      expect(rootSecond.result._tag).toBe('Completed')

      expect(rootTurns).toBeGreaterThanOrEqual(2)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('6) subagent yields without message → parent should be triggered', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      let rootTurns = 0

      yield* harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (script) =>
          script.setResolver(({ forkId }) => {
            if (forkId === null) {
              rootTurns += 1
              if (rootTurns === 1) {
                return {
                  xml: '<magnitude:invoke tool="agent_create">\n<magnitude:parameter name="agentId">baseline-sub-silent</magnitude:parameter>\n<magnitude:parameter name="type">explorer</magnitude:parameter>\n<magnitude:parameter name="title">baseline</magnitude:parameter>\n<magnitude:parameter name="message">do work</magnitude:parameter>\n</magnitude:invoke><magnitude:yield_user/>',
                }
              }
              return { xml: '<magnitude:yield_user/>' }
            }

            return { xml: '<magnitude:invoke tool="shell">\n<magnitude:parameter name="command">echo hello</magnitude:parameter>\n</magnitude:invoke><magnitude:yield_user/>' }
          }),
        ),
      )

      yield* harness.user('start subagent silent flow')

      const rootFirst = yield* harness.wait.turnCompleted(null)
      expect(rootFirst.result._tag).toBe('Completed')

      const created = yield* harness.wait.agentCreated((e) => e.agentId === 'baseline-sub-silent')
      expect(created.forkId).not.toBeNull()

      const subCompleted = yield* harness.wait.turnCompleted(created.forkId)
      expect(subCompleted.result._tag).toBe('Completed')

      let parentTriggered = false
      try {
        yield* harness.wait.event(
          'turn_completed',
          (e) => e.forkId === null && e.turnId !== rootFirst.turnId,
          { timeoutMs: 3000 },
        )
        parentTriggered = true
      } catch {
        parentTriggered = false
      }

      expect(parentTriggered).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
