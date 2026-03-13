import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import type { AppEvent } from '../../events'
import { createAgentTestHarness } from '../harness'
import { MockTurnScriptTag } from '../turn-script'

describe('subagent dynamics', () => {
  test('Orchestrator creates subagent', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.enqueue(
            {
              xml: `<actions><agent-create agentId="test-explorer"><type>explorer</type><title>test</title><message>do something</message></agent-create></actions><yield/>`,
            },
            null,
          ),
        ),
      )

      await harness.user('create a subagent')
      await harness.wait.turnCompleted()

      const created = await harness.wait.event(
        'agent_created',
        (e) => e.agentId === 'test-explorer' && e.role === 'explorer',
      )
      expect(created.type).toBe('agent_created')
      expect(created.forkId).not.toBeNull()
      expect(created.parentForkId).toBeNull()
      expect(created.name).toBe('test')
    } finally {
      await harness.dispose()
    }
  })

  test('Subagent turn can be scripted independently after creation', async () => {
    const harness = await createAgentTestHarness()
    try {
      await harness.runEffect(
        Effect.flatMap(MockTurnScriptTag, (s) =>
          s.setResolver(({ forkId }) => {
            if (forkId === null) {
              return {
                xml: `<actions><agent-create agentId="test-explorer"><type>explorer</type><title>test</title><message>do something</message></agent-create></actions><yield/>`,
              }
            }
            return {
              xml: '<comms><message to="parent">subagent done</message></comms><yield/>',
            }
          }),
        ),
      )

      await harness.user('create then run subagent')
      const rootCompleted = await harness.wait.turnCompleted(null)
      expect(rootCompleted.result.success).toBe(true)

      const created = await harness.wait.event('agent_created', (e) => e.agentId === 'test-explorer')
      const subCompleted = await harness.wait.turnCompleted(created.forkId)

      expect(subCompleted.type).toBe('turn_completed')
      expect(subCompleted.forkId).toBe(created.forkId)

      const hasSubagentTurn = harness
        .events()
        .some((e: AppEvent) => e.type === 'turn_started' && e.forkId === created.forkId)
      expect(hasSubagentTurn).toBe(true)
    } finally {
      await harness.dispose()
    }
  })
})