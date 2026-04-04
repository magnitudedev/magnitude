import { describe, test, expect } from 'bun:test'

/**
 * Tests the nonexistent agent ID validation logic used in execution-manager.ts.
 * This mirrors inline validation over resolved message targets:
 * filter non-user/non-parent targets, check against non-completed forks,
 * and flag invalid agent IDs.
 */

interface MinimalFork {
  agentId: string
  status: 'working' | 'completed'
}

function findInvalidAgentTargets(
  messagesSent: readonly { id: string; target: string }[],
  forks: Map<string, MinimalFork>
): string[] {
  const agentTargets = messagesSent.filter(m => m.target !== 'user' && m.target !== 'parent')
  if (agentTargets.length === 0) return []
  const knownAgentIds = new Set([...forks.values()].filter(f => f.status === 'working').map(f => f.agentId))
  return agentTargets.filter(m => !knownAgentIds.has(m.target)).map(m => m.target)
}

describe('nonexistent agent destination validation', () => {
  test('returns empty for messages to user/parent', () => {
    const messages = [
      { id: '1', target: 'user' },
      { id: '2', target: 'parent' },
    ]
    expect(findInvalidAgentTargets(messages, new Map())).toEqual([])
  })

  test('returns empty when agent exists and is working', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-explorer', status: 'working' as const }],
    ])
    const messages = [{ id: '1', target: 'my-explorer' }]
    expect(findInvalidAgentTargets(messages, forks)).toEqual([])
  })

  test('returns invalid dest when agent does not exist', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-explorer', status: 'working' as const }],
    ])
    const messages = [{ id: '1', target: 'nonexistent-agent' }]
    expect(findInvalidAgentTargets(messages, forks)).toEqual(['nonexistent-agent'])
  })

  test('returns invalid dest when agent exists but is completed', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-explorer', status: 'completed' as const }],
    ])
    const messages = [{ id: '1', target: 'my-explorer' }]
    expect(findInvalidAgentTargets(messages, forks)).toEqual(['my-explorer'])
  })

  test('returns multiple invalid dests', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-explorer', status: 'working' as const }],
    ])
    const messages = [
      { id: '1', target: 'bad-1' },
      { id: '2', target: 'my-explorer' },
      { id: '3', target: 'bad-2' },
    ]
    expect(findInvalidAgentTargets(messages, forks)).toEqual(['bad-1', 'bad-2'])
  })

  test('returns empty when no messages sent', () => {
    expect(findInvalidAgentTargets([], new Map())).toEqual([])
  })
})