import { describe, expect, test } from 'bun:test'
import type { SubagentTabItem } from '../components/chat/types'
import { reconcileForkMeta, sortSubagentTabs } from './use-subagent-tabs'

describe('sortSubagentTabs', () => {
  test('sorts active tabs before idle tabs, then by activeSince', () => {
    const tabs: SubagentTabItem[] = [
      { forkId: 'idle-older', agentId: 'idle-older', name: 'Idle Older', activeSince: 10, completedAt: 20, toolCount: 1, toolSummaryLine: 'x', statusLine: 'Agent is idle', phase: 'idle' },
      { forkId: 'active-newer', agentId: 'active-newer', name: 'Active Newer', activeSince: 30, toolCount: 1, toolSummaryLine: 'x', statusLine: 'Running…', phase: 'active' },
      { forkId: 'active-older', agentId: 'active-older', name: 'Active Older', activeSince: 5, toolCount: 1, toolSummaryLine: 'x', statusLine: 'Running…', phase: 'active' },
      { forkId: 'idle-newer', agentId: 'idle-newer', name: 'Idle Newer', activeSince: 40, completedAt: 50, toolCount: 1, toolSummaryLine: 'x', statusLine: 'Agent is idle', phase: 'idle' },
    ]

    const sorted = [...tabs].sort(sortSubagentTabs)
    expect(sorted.map(tab => tab.forkId)).toEqual(['active-older', 'active-newer', 'idle-older', 'idle-newer'])
  })
})

describe('reconcileForkMeta', () => {
  test('does not synthesize completedAt on active -> idle transition when completedAt is missing', () => {
    const running = new Map<string, any>([
      ['fork-1', { forkId: 'fork-1', name: 'A', activeSince: 1000, status: 'running', toolCounts: {} }],
    ])
    const first = reconcileForkMeta({
      prev: {},
      latestByFork: running,
      agentStatusState: null,
      now: 5000,
    })
    expect(first.next['fork-1']?.phase).toBe('active')
    expect(first.next['fork-1']?.completedAt).toBeUndefined()

    const completed = new Map<string, any>([
      ['fork-1', { forkId: 'fork-1', name: 'A', activeSince: 1000, status: 'completed', toolCounts: {} }],
    ])
    const second = reconcileForkMeta({
      prev: first.next,
      latestByFork: completed,
      agentStatusState: null,
      now: 9000,
    })
    expect(second.next['fork-1']?.phase).toBe('idle')
    expect(second.next['fork-1']?.completedAt).toBeUndefined()

    const third = reconcileForkMeta({
      prev: second.next,
      latestByFork: completed,
      agentStatusState: null,
      now: 12000,
    })
    expect(third.next['fork-1']?.phase).toBe('idle')
    expect(third.next['fork-1']?.completedAt).toBeUndefined()
  })

  test('restored idle without completedAt preserves prior completedAt (no now fallback freeze)', () => {
    const latestByFork = new Map<string, any>([
      ['fork-1', { forkId: 'fork-1', name: 'A', activeSince: 1000, status: 'completed', toolCounts: {} }],
    ])

    const result = reconcileForkMeta({
      prev: {
        'fork-1': {
          agentId: 'a',
          name: 'A',
          activeSince: 1000,
          toolCount: 0,
          toolCounts: {},
          phase: 'idle' as const,
          completedAt: 7000,
        },
      },
      latestByFork,
      agentStatusState: null,
      now: 9000,
    })

    expect(result.next['fork-1']?.phase).toBe('idle')
    expect(result.next['fork-1']?.completedAt).toBe(7000)
  })

  test('derives phase from fork activity status (not agent status) and preserves activity completedAt', () => {
    const latestByFork = new Map<string, any>([
      ['fork-1', { forkId: 'fork-1', name: 'A', activeSince: 1000, status: 'completed', completedAt: 7000, toolCounts: {} }],
    ])
    const agentStatusState = {
      agents: new Map([
        ['a', { agentId: 'a', forkId: 'fork-1', status: 'working' }],
      ]),
    } as any

    const result = reconcileForkMeta({
      prev: {
        'fork-1': {
          agentId: 'a',
          name: 'A',
          activeSince: 1000,
          toolCount: 0,
          toolCounts: {},
          phase: 'idle' as const,
          completedAt: 6000,
        },
      },
      latestByFork,
      agentStatusState,
      now: 9000,
    })

    expect(result.next['fork-1']?.phase).toBe('idle')
    expect(result.next['fork-1']?.completedAt).toBe(7000)
  })

  test('clears completedAt when status returns to running', () => {
    const latestByFork = new Map<string, any>([
      ['fork-1', { forkId: 'fork-1', name: 'A', activeSince: 1000, status: 'running', toolCounts: {} }],
    ])

    const result = reconcileForkMeta({
      prev: {
        'fork-1': {
          agentId: 'a',
          name: 'A',
          activeSince: 1000,
          toolCount: 0,
          toolCounts: {},
          phase: 'idle' as const,
          completedAt: 7000,
        },
      },
      latestByFork,
      agentStatusState: null,
      now: 9000,
    })

    expect(result.next['fork-1']?.phase).toBe('active')
    expect(result.next['fork-1']?.completedAt).toBeUndefined()
  })

  test('marks dismissed or nonexistent prior forks for prune', () => {
    const prev = {
      'fork-dismissed': {
        agentId: 'a',
        name: 'A',
        activeSince: 1000,
        toolCount: 0,
        toolCounts: {},
        phase: 'idle' as const,
        completedAt: 2000,
      },
      'fork-missing': {
        agentId: 'b',
        name: 'B',
        activeSince: 1000,
        toolCount: 0,
        toolCounts: {},
        phase: 'idle' as const,
        completedAt: 2000,
      },
    }

    const agentStatusState = {
      agents: new Map([
        ['a', { agentId: 'a', forkId: 'fork-dismissed', status: 'dismissed' }],
      ]),
    } as any

    const result = reconcileForkMeta({
      prev,
      latestByFork: new Map(),
      agentStatusState,
      now: 6000,
    })

    expect(result.pruneForkIds.sort()).toEqual(['fork-dismissed', 'fork-missing'])
    expect(Object.keys(result.next)).toEqual([])
  })
})