import { describe, expect, test } from 'bun:test'
import type { AgentCreated } from '../../events'
import type { ForkActivityMessage } from '../../projections/display'


describe('subagent timer semantics', () => {
  test('fork activity timing fields support active-stint semantics', () => {
    const createdEvent: AgentCreated = {
      type: 'agent_created',
      forkId: 'fork-sub',
      parentForkId: null,
      agentId: 'agent-sub',
      role: 'builder',
      name: 'Builder',
      context: 'ctx',
      mode: 'spawn',
      taskId: 'task-1',
      message: '',
    }

    const created: ForkActivityMessage = {
      id: 'm1',
      type: 'fork_activity',
      forkId: createdEvent.forkId,
      name: createdEvent.name,
      role: createdEvent.role,
      status: 'running',
      createdAt: 1000,
      activeSince: 1000,
      accumulatedActiveMs: 0,
      resumeCount: 0,
      toolCounts: {
        commands: 0, reads: 0, writes: 0, edits: 0, searches: 0, webSearches: 0, webFetches: 0,
        artifactWrites: 0, artifactUpdates: 0, clicks: 0, navigations: 0, inputs: 0, evaluations: 0, other: 0
      },
      artifactNames: [],
      timestamp: 1000,
    }

    const paused: ForkActivityMessage = {
      ...created,
      status: 'completed',
      completedAt: 3000,
      accumulatedActiveMs: 2000,
    }

    const resumed: ForkActivityMessage = {
      ...paused,
      status: 'running',
      activeSince: 4000,
      completedAt: undefined,
      resumeCount: 1,
      timestamp: 4000,
    }

    expect(created.createdAt).toBe(created.activeSince)
    expect(paused.completedAt).toBeDefined()
    expect((paused.completedAt ?? 0)).toBeGreaterThanOrEqual(created.activeSince)
    expect(resumed.activeSince).toBeGreaterThan(paused.completedAt ?? 0)
    expect(resumed.createdAt).toBe(created.createdAt)
    expect(resumed.completedAt).toBeUndefined()
    expect(created.accumulatedActiveMs).toBe(0)
    expect(paused.accumulatedActiveMs).toBe(2000)
    expect(resumed.accumulatedActiveMs).toBe(2000)

    const pausedAgain: ForkActivityMessage = {
      ...resumed,
      status: 'completed',
      completedAt: 5500,
      accumulatedActiveMs: resumed.accumulatedActiveMs + (5500 - resumed.activeSince),
    }
    expect(pausedAgain.accumulatedActiveMs).toBe(3500)
  })


})