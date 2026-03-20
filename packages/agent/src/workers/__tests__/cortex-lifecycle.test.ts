import { describe, test, expect } from 'bun:test'

describe('Cortex fork lifecycle parity', () => {
  test('configured via source: activate on agent_created and complete on agent_killed + subagent_user_killed + subagent_idle_closed', async () => {
    const source = await Bun.file('packages/agent/src/workers/cortex.ts').text()
    expect(source.includes("activateOn: 'agent_created'")).toBe(true)
    expect(source.includes("completeOn: ['agent_killed', 'subagent_user_killed', 'subagent_idle_closed']")).toBe(true)
    expect(source.includes('subagent_idle_closed: (event)')).toBe(true)
  })
})
