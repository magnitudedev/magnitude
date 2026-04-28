import { describe, expect, it } from '@effect/vitest'
import { Effect, Exit } from 'effect'
import { YIELD_USER } from '@magnitudedev/xml-act'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { MockTurnScriptTag } from '../src/test-harness/turn-script'
import { response } from '../src/test-harness/response-builder'

const spawnWorkerXml = (id: string, title: string, message: string) =>
  response().createTask(id, title).spawnWorker(id, message).yield().xml!

describe('parent wake edge cases', () => {
  it.live('parent woken when subagent has unexpected error', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      let rootTurns = 0

      yield* h.script.setResolver(({ forkId }) => {
        if (forkId === null) {
          rootTurns += 1
          if (rootTurns === 1) {
            return { xml: spawnWorkerXml('error-sub', 'err', 'go') }
          }
          return { xml: YIELD_USER }
        }
        // Subagent resolver throws -> triggers turn_outcome
        throw new Error('simulated subagent crash')
      })

      yield* h.user('start')

      const rootFirst = yield* h.wait.turnCompleted(null)
      expect(rootFirst.outcome._tag).toBe('Completed')

      const created = yield* h.wait.agentCreated((e) => e.agentId === 'error-sub')

      // Subagent should get turn_outcome
      const subError = yield* h.wait.event(
        'turn_outcome',
        (e) => e.forkId === created.forkId,
      )
      expect(subError.outcome._tag === 'UnexpectedError' ? subError.outcome.message : '').toContain('simulated subagent crash')

      // Parent should be woken
      const parentWake = yield* Effect.exit(
        h.wait.event(
          'turn_outcome',
          (e) => e.forkId === null && e.turnId !== rootFirst.turnId,
          { timeoutMs: 3000 },
        ),
      )

      expect(Exit.isSuccess(parentWake)).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('parent NOT woken when already-idle subagent is user-killed', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      let rootTurns = 0

      yield* h.script.setResolver(({ forkId }) => {
        if (forkId === null) {
          rootTurns += 1
          if (rootTurns === 1) {
            return { xml: spawnWorkerXml('kill-sub', 'will be killed', 'do work') }
          }
          return { xml: YIELD_USER }
        }

        return { xml: `<magnitude:message to="parent">subagent done</magnitude:message>\n${YIELD_USER}` }
      })

      yield* h.user('start killable subagent')

      const rootFirst = yield* h.wait.turnCompleted(null)
      expect(rootFirst.outcome._tag).toBe('Completed')

      const created = yield* h.wait.agentCreated((e) => e.agentId === 'kill-sub')
      yield* h.wait.turnCompleted(created.forkId)

      const rootSecond = yield* h.wait.event(
        'turn_outcome',
        (e) => e.forkId === null && e.turnId !== rootFirst.turnId,
      )

      // Now root is idle. Kill the already-idle subagent.
      yield* h.send({
        type: 'subagent_user_killed',
        forkId: created.forkId,
        parentForkId: null,
        agentId: 'kill-sub',
        source: 'tab_close_confirm',
      })

      // Parent should NOT be woken — killing idle subagent is just cleanup
      const parentWake = yield* Effect.exit(
        h.wait.event(
          'turn_outcome',
          (e) => e.forkId === null && e.turnId !== rootFirst.turnId && e.turnId !== rootSecond.turnId,
          { timeoutMs: 3000 },
        ),
      )

      expect(Exit.isFailure(parentWake)).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
