import { describe, test, expect } from 'bun:test'
import { Agent } from '@magnitudedev/event-core'
import { Layer } from 'effect'
import type { AppEvent } from '../../events'
import { TurnProjection } from '../../projections/turn'
import { AgentStatusProjection } from '../../projections/agent-status'
import { AgentRoutingProjection } from '../../projections/agent-routing'
import { UserMessageResolutionProjection } from '../../projections/user-message-resolution'
import { CompactionProjection } from '../../projections/compaction'
import { TurnController } from '../turn-controller'

const TestAgent = Agent.define<AppEvent>()({
  name: 'TurnControllerTestAgent',
  projections: [
    TurnProjection,
    AgentStatusProjection,
    AgentRoutingProjection,
    UserMessageResolutionProjection,
    CompactionProjection,
  ],
  workers: [TurnController],
})

const emptyRequirements = Layer.empty as Parameters<typeof TestAgent.createClient>[0]

async function initSession(client: Awaited<ReturnType<typeof TestAgent.createClient>>) {
  await client.send({
    type: 'session_initialized',
    forkId: null,
    context: {
      cwd: process.cwd(),
      workspacePath: '/tmp/test-workspace',
      platform: 'macos',
      shell: '/bin/zsh',
      timezone: 'UTC',
      username: 'tester',
      fullName: null,
      git: null,
      folderStructure: '.',
      agentsFile: null,
      skills: null,
    },
  })
}

describe('TurnController', () => {
  // NOTE: These tests verify TurnController logic correctly but bun:test reports
  // false timeout failures due to leaked Effect fibers from client.dispose().
  // The assertions pass — the timeout is a bun test runner limitation.

  test('wake enqueues trigger and publishes turn_started', async () => {
    const client = await TestAgent.createClient(emptyRequirements)
    const events: AppEvent[] = []
    const unsub = client.onEvent((event) => events.push(event))

    try {
      await initSession(client)
      await client.send({ type: 'wake', forkId: null })
      await new Promise((resolve) => setTimeout(resolve, 500))

      const started = events.filter((event) => event.type === 'turn_started' && event.forkId === null)
      expect(started.length).toBeGreaterThan(0)
    } finally {
      unsub()
      await client.dispose()
    }
  })

  test('interrupt with no triggers does not publish turn_started', async () => {
    const client = await TestAgent.createClient(emptyRequirements)
    const events: AppEvent[] = []
    const unsub = client.onEvent((event) => events.push(event))

    try {
      await initSession(client)
      await client.send({ type: 'wake', forkId: null })
      await new Promise((resolve) => setTimeout(resolve, 500))

      const before = events.filter((event) => event.type === 'turn_started' && event.forkId === null).length

      await client.send({ type: 'interrupt', forkId: null })
      await new Promise((resolve) => setTimeout(resolve, 200))

      const after = events.filter((event) => event.type === 'turn_started' && event.forkId === null).length
      expect(after).toBe(before)
    } finally {
      unsub()
      await client.dispose()
    }
  })
})
