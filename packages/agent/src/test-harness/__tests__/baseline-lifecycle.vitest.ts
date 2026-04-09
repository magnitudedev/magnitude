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
        xml: '<message to="user">done</message><idle/>',
      })

      yield* harness.user('hello')
      const completed = yield* harness.wait.turnCompleted(null)

      expect(completed.result.success).toBe(true)
      if (completed.result.success) {
        expect(completed.result.turnDecision).toBe('idle')
      }

      const root = yield* harness.projectionFork(TurnProjection.Tag, null)
      expect(root._tag).toBe('idle')
      expect(root.triggers.length).toBe(0)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('3) multi-turn chain (next then yield)', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: '<shell>echo chain</shell>' })
      yield* harness.script.next({ xml: '<idle/>' })

      yield* harness.user('run chain')

      const first = yield* harness.wait.turnCompleted(null)
      const second = yield* harness.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId,
      )

      expect(first.result.success).toBe(true)
      if (first.result.success) {
        expect(first.result.turnDecision).toBe('continue')
      }

      expect(second.result.success).toBe(true)
      if (second.result.success) {
        expect(second.result.turnDecision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('4) tool execution in a turn (write)', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({
        xml: '<write path="output.txt">content</write><idle/>',
      })

      yield* harness.user('write a file')
      const completed = yield* harness.wait.turnCompleted(null)
      const toolEnded = yield* harness.wait.event(
        'tool_event',
        (e) => e.forkId === null && e.event._tag === 'ToolExecutionEnded' && e.event.toolName === 'write',
      )

      expect(completed.result.success).toBe(true)
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
                  xml: '<agent-create id="baseline-sub" type="explorer"><title>baseline</title><message>do work</message></agent-create><idle/>',
                }
              }
              return { xml: '<idle/>' }
            }

            return { xml: '<message to="parent">subagent done</message><idle/>' }
          }),
        ),
      )

      yield* harness.user('start subagent flow')

      const rootFirst = yield* harness.wait.turnCompleted(null)
      expect(rootFirst.result.success).toBe(true)

      const created = yield* harness.wait.agentCreated((e) => e.agentId === 'baseline-sub')
      expect(created.forkId).not.toBeNull()

      const subCompleted = yield* harness.wait.turnCompleted(created.forkId)
      expect(subCompleted.result.success).toBe(true)

      const rootSecond = yield* harness.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== rootFirst.turnId,
      )
      expect(rootSecond.result.success).toBe(true)

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
                  xml: '<agent-create id="baseline-sub-silent" type="explorer"><title>baseline</title><message>do work</message></agent-create><idle/>',
                }
              }
              return { xml: '<idle/>' }
            }

            return { xml: '<shell>echo hello</shell><idle/>' }
          }),
        ),
      )

      yield* harness.user('start subagent silent flow')

      const rootFirst = yield* harness.wait.turnCompleted(null)
      expect(rootFirst.result.success).toBe(true)

      const created = yield* harness.wait.agentCreated((e) => e.agentId === 'baseline-sub-silent')
      expect(created.forkId).not.toBeNull()

      const subCompleted = yield* harness.wait.turnCompleted(created.forkId)
      expect(subCompleted.result.success).toBe(true)

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
