import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { MockTurnScriptTag } from '../src/test-harness/turn-script'
import { response } from '../src/test-harness/response-builder'
import type { AppEvent } from '../src/events'

const spawnXml = (id: string, title: string, message: string) =>
  response().createTask(id, title).spawnWorker(id, message).yield().xml!

describe('subagent dynamics', () => {
  it.live('Orchestrator creates subagent', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xml: spawnXml('test-explorer', 'test', 'do something'),
            },
            null,
          ),
        ),
      )

      yield* h.user('create a subagent')
      yield* h.wait.turnCompleted(null)

      const created = yield* h.wait.event(
        'agent_created',
        (e) => e.agentId === 'test-explorer' && e.role === 'worker',
      )

      expect(created.type).toBe('agent_created')
      expect(created.forkId).not.toBeNull()
      expect(created.parentForkId).toBeNull()
      expect(created.name).toBe('test')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Subagent turn can be scripted independently after creation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.setResolver(({ forkId }) => {
            if (forkId === null) {
              return {
                xml: spawnXml('test-explorer', 'test', 'do something'),
              }
            }

            return {
              xml: '<magnitude:message to="parent">subagent done</magnitude:message><magnitude:yield_user/>',
            }
          }),
        ),
      )

      yield* h.user('create then run subagent')
      const rootCompleted = yield* h.wait.turnCompleted(null)
      expect(rootCompleted.outcome._tag).toBe('Completed')

      const created = yield* h.wait.event('agent_created', (e) => e.agentId === 'test-explorer')
      const subCompleted = yield* h.wait.turnCompleted(created.forkId)

      expect(subCompleted.type).toBe('turn_outcome')
      expect(subCompleted.forkId).toBe(created.forkId)

      const hasSubagentTurn = h
        .events()
        .some((e: AppEvent) => e.type === 'turn_started' && e.forkId === created.forkId)
      expect(hasSubagentTurn).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
