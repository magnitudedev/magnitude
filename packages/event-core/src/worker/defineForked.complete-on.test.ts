import { describe, test, expect } from 'bun:test'
import { Effect } from 'effect'
import { make as makeEventEngine } from '../event-engine'
import { defineForked } from './defineForked'
import type { ForkableEvent } from '../projection/defineForked'

type TestEvent =
  | { type: 'agent_created'; forkId: string | null }
  | { type: 'agent_killed'; forkId: string | null }
  | { type: 'worker_user_killed'; forkId: string | null }
  | { type: 'turn_started'; forkId: string | null }

describe('Worker.defineForked completeOn lifecycle', () => {
  test('tears down fork worker on single completeOn and prevents post-kill execution', async () => {
    const seen: Array<{ type: string; forkId: string | null }> = []

    const ForkedWorker = defineForked<TestEvent & ForkableEvent>()({
      name: 'ForkedWorkerTest',
      forkLifecycle: {
        activateOn: 'agent_created',
        completeOn: 'agent_killed',
      },
      eventHandlers: {
        turn_started: (event) => {
          seen.push({ type: event.type, forkId: event.forkId })
          return Effect.void
        },
      },
    })

    const TestAgent = makeEventEngine<TestEvent & ForkableEvent>()({
      name: 'ForkedWorkerLifecycleAgent',
      schemaVersion: 'test',
      projections: [],
      workers: [ForkedWorker],
    })

    const client = await TestAgent.createClient()
    try {
      await client.send({ type: 'agent_created', forkId: 'fork-a' })
      await client.send({ type: 'turn_started', forkId: 'fork-a' })
      await client.send({ type: 'agent_killed', forkId: 'fork-a' })
      await client.send({ type: 'turn_started', forkId: 'fork-a' })

      await new Promise((r) => setTimeout(r, 30))

      const forForkA = seen.filter((e) => e.forkId === 'fork-a')
      expect(forForkA.length).toBe(1)
      expect(forForkA[0]?.type).toBe('turn_started')
    } finally {
      await client.dispose()
    }
  })

  test('tears down fork worker when any completeOn event in list is published', async () => {
    const seen: Array<{ type: string; forkId: string | null }> = []

    const ForkedWorker = defineForked<TestEvent & ForkableEvent>()({
      name: 'ForkedWorkerTest',
      forkLifecycle: {
        activateOn: 'agent_created',
        completeOn: ['agent_killed', 'worker_user_killed'],
      },
      eventHandlers: {
        turn_started: (event) => {
          seen.push({ type: event.type, forkId: event.forkId })
          return Effect.void
        },
      },
    })

    const TestAgent = makeEventEngine<TestEvent & ForkableEvent>()({
      name: 'ForkedWorkerLifecycleAgent',
      schemaVersion: 'test',
      projections: [],
      workers: [ForkedWorker],
    })

    const client = await TestAgent.createClient()
    try {
      await client.send({ type: 'agent_created', forkId: 'fork-a' })
      await client.send({ type: 'turn_started', forkId: 'fork-a' })
      await client.send({ type: 'worker_user_killed', forkId: 'fork-a' })
      await client.send({ type: 'turn_started', forkId: 'fork-a' })

      await new Promise((r) => setTimeout(r, 30))

      const forForkA = seen.filter((e) => e.forkId === 'fork-a')
      expect(forForkA.length).toBe(1)
      expect(forForkA[0]?.type).toBe('turn_started')
    } finally {
      await client.dispose()
    }
  })
})
