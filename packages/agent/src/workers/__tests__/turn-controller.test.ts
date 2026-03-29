import { describe, test, expect } from 'bun:test'
import { Agent } from '@magnitudedev/event-core'
import { Layer } from 'effect'
import type { AppEvent } from '../../events'
import { WorkingStateProjection } from '../../projections/working-state'
import { AgentStatusProjection } from '../../projections/agent-status'
import { TurnController } from '../turn-controller'
import { createId } from '../../util/id'

const TestAgent = Agent.define<AppEvent>()({
  name: 'TurnControllerTestAgent',
  projections: [WorkingStateProjection, AgentStatusProjection],
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
      userMemory: null,
    },
  })
}

async function waitFor(
  predicate: () => boolean,
  timeoutMs = 500,
  intervalMs = 10,
): Promise<void> {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    if (predicate()) return
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
  throw new Error('Timed out waiting for condition')
}

describe('TurnController', () => {
  test('active subagent publishes turn_started', async () => {
    const client = await TestAgent.createClient(emptyRequirements)
    const events: AppEvent[] = []
    const unsub = client.onEvent((event) => events.push(event))

    try {
      await initSession(client)

      await client.send({
        type: 'agent_created',
        forkId: 'fork-active',
        parentForkId: null,
        agentId: 'agent-active',
        name: 'Active Agent',
        role: 'builder',
        mode: 'spawn',
        context: '',
        taskId: 'task-active',
        message: 'start',
      })

      // Force a deterministic false -> true shouldTrigger transition after agent exists
      await client.send({ type: 'interrupt', forkId: 'fork-active' })
      await client.send({ type: 'wake', forkId: 'fork-active' })

      await waitFor(() =>
        events.some((event) => event.type === 'turn_started' && event.forkId === 'fork-active')
      )

      const started = events.filter((event) => event.type === 'turn_started' && event.forkId === 'fork-active')
      expect(started.length).toBeGreaterThan(0)
    } finally {
      unsub()
      await client.dispose()
    }
  })

  test('killed/nonexistent subagent can still publish turn_started (execution protection is completeOn lifecycle)', async () => {
    const client = await TestAgent.createClient(emptyRequirements)
    const events: AppEvent[] = []
    const unsub = client.onEvent((event) => events.push(event))

    try {
      await initSession(client)

      await client.send({
        type: 'agent_created',
        forkId: 'fork-killed',
        parentForkId: null,
        agentId: 'agent-killed',
        name: 'Killed Agent',
        role: 'builder',
        mode: 'spawn',
        context: '',
        taskId: 'task-killed',
        message: 'start',
      })

      await waitFor(() =>
        events.some((event) => event.type === 'turn_started' && event.forkId === 'fork-killed')
      )

      const beforeKillCount = events.filter(
        (event) => event.type === 'turn_started' && event.forkId === 'fork-killed'
      ).length

      await client.send({
        type: 'agent_killed',
        forkId: 'fork-killed',
        parentForkId: null,
        agentId: 'agent-killed',
        reason: 'test kill',
      })

      // Clear working/willContinue so a subsequent wake can retrigger shouldTriggerChanged
      await client.send({ type: 'interrupt', forkId: 'fork-killed' })

      await client.send({ type: 'wake', forkId: 'fork-killed' })
      await client.send({ type: 'wake', forkId: 'fork-ghost' })

      await new Promise((resolve) => setTimeout(resolve, 50))

      await waitFor(() =>
        events.some((event) => event.type === 'turn_started' && event.forkId === 'fork-killed')
          && events.some((event) => event.type === 'turn_started' && event.forkId === 'fork-ghost')
      )

      const killedAfter = events.filter(
        (event) => event.type === 'turn_started' && event.forkId === 'fork-killed'
      ).length
      const ghostAfter = events.filter(
        (event) => event.type === 'turn_started' && event.forkId === 'fork-ghost'
      ).length

      expect(killedAfter).toBeGreaterThan(beforeKillCount)
      expect(ghostAfter).toBeGreaterThan(0)
    } finally {
      unsub()
      await client.dispose()
    }
  })

  test('root still publishes turn_started', async () => {
    const client = await TestAgent.createClient(emptyRequirements)
    const events: AppEvent[] = []
    const unsub = client.onEvent((event) => events.push(event))

    try {
      await initSession(client)

      await client.send({
        type: 'user_message',
        messageId: createId(),
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'trigger root turn' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })

      // Force deterministic false -> true shouldTrigger transition for root
      await client.send({ type: 'interrupt', forkId: null })
      await client.send({ type: 'wake', forkId: null })

      await waitFor(() =>
        events.some((event) => event.type === 'turn_started' && event.forkId === null)
      )

      const started = events.filter((event) => event.type === 'turn_started' && event.forkId === null)
      expect(started.length).toBeGreaterThan(0)
    } finally {
      unsub()
      await client.dispose()
    }
  })

  test('shouldTrigger=false no-op', async () => {
    const client = await TestAgent.createClient(emptyRequirements)
    const events: AppEvent[] = []
    const unsub = client.onEvent((event) => events.push(event))

    try {
      await initSession(client)

      await client.send({ type: 'wake', forkId: null })
      await waitFor(() =>
        events.some((event) => event.type === 'turn_started' && event.forkId === null)
      )

      const before = events.filter((event) => event.type === 'turn_started' && event.forkId === null).length

      // interrupt drives shouldTrigger=false; TurnController must no-op for false
      await client.send({ type: 'interrupt', forkId: null })
      await new Promise((resolve) => setTimeout(resolve, 50))

      const after = events.filter((event) => event.type === 'turn_started' && event.forkId === null).length
      expect(after).toBe(before)
    } finally {
      unsub()
      await client.dispose()
    }
  })
})
