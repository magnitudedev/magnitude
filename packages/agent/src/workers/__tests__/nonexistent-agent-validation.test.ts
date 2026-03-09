import { describe, test, expect } from 'bun:test'

/**
 * Tests the nonexistent agent ID validation logic used in execution-manager.ts at ExecutionEnd.
 * This mirrors the inline validation: filter sentMessages for non-user/non-parent dests,
 * check against running forks, flag invalid destinations.
 */

interface MinimalFork {
  agentId: string
  status: 'running' | 'completed'
}

function findInvalidAgentDests(
  messagesSent: readonly { id: string; dest: string }[],
  forks: Map<string, MinimalFork>
): string[] {
  const agentDests = messagesSent.filter(m => m.dest !== 'user' && m.dest !== 'parent')
  if (agentDests.length === 0) return []
  const knownAgentIds = new Set([...forks.values()].filter(f => f.status === 'running').map(f => f.agentId))
  return agentDests.filter(m => !knownAgentIds.has(m.dest)).map(m => m.dest)
}

describe('nonexistent agent destination validation', () => {
  test('returns empty for messages to user/parent', () => {
    const messages = [
      { id: '1', dest: 'user' },
      { id: '2', dest: 'parent' },
    ]
    expect(findInvalidAgentDests(messages, new Map())).toEqual([])
  })

  test('returns empty when agent exists and is running', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-scout', status: 'running' as const }],
    ])
    const messages = [{ id: '1', dest: 'my-scout' }]
    expect(findInvalidAgentDests(messages, forks)).toEqual([])
  })

  test('returns invalid dest when agent does not exist', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-scout', status: 'running' as const }],
    ])
    const messages = [{ id: '1', dest: 'nonexistent-agent' }]
    expect(findInvalidAgentDests(messages, forks)).toEqual(['nonexistent-agent'])
  })

  test('returns invalid dest when agent exists but is completed', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-scout', status: 'completed' as const }],
    ])
    const messages = [{ id: '1', dest: 'my-scout' }]
    expect(findInvalidAgentDests(messages, forks)).toEqual(['my-scout'])
  })

  test('returns multiple invalid dests', () => {
    const forks = new Map([
      ['fork-1', { agentId: 'my-scout', status: 'running' as const }],
    ])
    const messages = [
      { id: '1', dest: 'bad-1' },
      { id: '2', dest: 'my-scout' },
      { id: '3', dest: 'bad-2' },
    ]
    expect(findInvalidAgentDests(messages, forks)).toEqual(['bad-1', 'bad-2'])
  })

  test('returns empty when no messages sent', () => {
    expect(findInvalidAgentDests([], new Map())).toEqual([])
  })
})