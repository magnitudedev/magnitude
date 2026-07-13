import { describe, test, expect } from 'bun:test'

describe('Cortex fork lifecycle parity', () => {
  test('configured via source: activate on agent_created and complete on agent_killed + worker_user_killed + worker_idle_closed', async () => {
    const source = await Bun.file('packages/agent/src/workers/cortex.ts').text()
    expect(source.includes("activateOn: 'agent_created'")).toBe(true)
    expect(source.includes("completeOn: ['agent_killed', 'worker_user_killed', 'worker_idle_closed']")).toBe(true)
    expect(source.includes('worker_idle_closed: (event)')).toBe(true)
  })
})
