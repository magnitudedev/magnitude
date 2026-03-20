import { describe, test, expect } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

describe('AgentOrchestrator lifecycle wiring', () => {
  test('subagent_user_killed disposes fork and wakes parent fork', () => {
    const source = readFileSync(join(import.meta.dir, '..', 'agent-orchestrator.ts'), 'utf8')
    expect(source.includes('subagent_user_killed: (event, publish)')).toBe(true)
    expect(source.includes('yield* execManager.disposeFork(event.forkId)')).toBe(true)
    expect(source.includes("yield* publish({ type: 'wake', forkId: event.parentForkId })")).toBe(true)
  })
})
